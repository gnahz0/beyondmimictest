import time
import threading
import argparse
from pathlib import Path
import numpy as np
import csv
import os

# --- Vuer env ---
from vuer_sim_example.sim.vuer_sim import VuerSim
from vuer_sim_example.envs.base_env import BaseEnv
from vuer_sim_example.tasks.falcon_task import FalconTask

import xml.etree.ElementTree as ET
# --- ONNX policy adapter (BeyondMimic) ---
import onnx, onnxruntime as ort
import numpy as np

FREE_BASE_QPOS = 7  # free base in qpos: quat(4) + pos(3)
FREE_BASE_QVEL = 6  # free base in qvel: ang(3) + lin(3)

def get_mj_actuator_joint_names_recursive(xml_path: str) -> list[str]:
    xml_path = os.path.abspath(xml_path)
    seen = set()
    out = []

    def _walk(path):
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            return
        seen.add(path)

        tree = ET.parse(path)
        root = tree.getroot()
        base_dir = os.path.dirname(path)

        # collect actuator → joint names in the order they appear
        for elem in root.findall(".//actuator/*"):
            j = elem.get("joint")
            if j:
                out.append(j)

        # follow nested includes
        for inc in root.findall(".//include"):
            inc_file = inc.get("file")
            if not inc_file:
                continue
            inc_path = inc_file if os.path.isabs(inc_file) else os.path.join(base_dir, inc_file)
            _walk(inc_path)

    _walk(xml_path)
    return out

def quat_to_mat(q, fmt="wxyz"):
    q = np.asarray(q, dtype=np.float32)
    if fmt == "xyzw":
        x, y, z, w = q
    else:  # "wxyz"
        w, x, y, z = q
    # normalize
    n = np.linalg.norm([w, x, y, z]) + 1e-8
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


class MotionRef:
    """Loads motion.npz and provides command/anchor at timestep t."""
    def __init__(self, npz_path: str, n: int, anchor_body_name: str | None, body_names: list[str] | None):
        self.data = np.load(npz_path)
        self.n = n

        # Required arrays present in your file:
        # joint_pos [T, n], joint_vel [T, n]
        # body_pos_w [T, B, 3], body_quat_w [T, B, 4]
        try:
            self.jpos = np.asarray(self.data["joint_pos"], dtype=np.float32)
            self.jvel = np.asarray(self.data["joint_vel"], dtype=np.float32)
        except KeyError as e:
            raise KeyError(f"motion.npz missing {e.args[0]} (expected 'joint_pos' and 'joint_vel')")

        self.body_pos_w = np.asarray(self.data["body_pos_w"], dtype=np.float32)  # [T,B,3]
        self.body_quat_w = np.asarray(self.data["body_quat_w"], dtype=np.float32)  # [T,B,4]

        self.body_lin_vel_w = np.asarray(self.data["body_lin_vel_w"], dtype=np.float32)  # [T,B,3]
        self.body_ang_vel_w = np.asarray(self.data["body_ang_vel_w"], dtype=np.float32)  # [T,B,3]

        # time length checks
        self.T = self.jpos.shape[0]
        assert self.jvel.shape[0] == self.T and self.body_pos_w.shape[0] == self.T and self.body_quat_w.shape[0] == self.T, \
            "All motion arrays must share the same time length T."

        # Figure out which body index is the anchor
        if body_names and anchor_body_name and anchor_body_name in body_names:
            self.anchor_idx = body_names.index(anchor_body_name)
        else:
            # fallback: first body (often root)
            self.anchor_idx = 0

        # light shape sanity
        assert self.jpos.shape[1] == self.n, f"joint_pos second dim {self.jpos.shape[1]} != n ({self.n})"
        assert self.jvel.shape[1] == self.n, f"joint_vel second dim {self.jvel.shape[1]} != n ({self.n})"
        assert self.body_pos_w.shape[2] == 3 and self.body_quat_w.shape[2] == 4, "bad body_pos/body_quat shapes"
        self.t = 0
        print(f"[MotionRef] Loaded {npz_path} (T={self.T}, n={self.n}, anchor='{anchor_body_name}' idx={self.anchor_idx})")
    
    def step(self):
        self.t = (self.t + 1) % self.T

    def anchor_rot_w(self):
        q = self.body_quat_w[self.t, self.anchor_idx, :]
        return quat_to_mat(q)
    def anchor_pos_w(self):
        return self.body_pos_w[self.t, self.anchor_idx, :]

class BMOnnxPolicy:
    def __init__(self, onnx_path: str, motion_npz_path: str, mj_joint_names: list[str] | None = None):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.obs_dim  = int(self.sess.get_inputs()[0].shape[1])  # expect 160
        self.out_name = self.sess.get_outputs()[0].name

        m  = onnx.load(str(onnx_path))
        md = {p.key: p.value for p in m.metadata_props}

        def _csv(s):    return [x.strip() for x in s.split(",")] if s else []
        def _floats(s): return np.array([float(x) for x in s.split(",") if x.strip()!=""], dtype=np.float32) if s else None

        # required metadata
        self.obs_names   = _csv(md.get("observation_names",""))     # <- we will honor this order
        self.meta_jnames = _csv(md.get("joint_names",""))
        self.n = len(self.meta_jnames)
        self.command_dim = 2 * self.n
        assert self.n > 0, "ONNX missing joint_names metadata."

        aS = _floats(md.get("action_scale",""))
        dJ = _floats(md.get("default_joint_pos",""))
        self.action_scale      = aS if (aS is not None and aS.size == self.n) else np.ones(self.n, np.float32)
        self.default_joint_pos = dJ if (dJ is not None and dJ.size == self.n) else np.zeros(self.n, np.float32)

        # optional
        self.joint_stiffness     = _floats(md.get("joint_stiffness",""))
        self.joint_damping       = _floats(md.get("joint_damping",""))
        self.motor_effort_limits = _floats(md.get("motor_effort_limits",""))

        # mapping meta -> mj
        self.meta_to_mj = None
        if mj_joint_names and len(mj_joint_names) >= self.n:
            mj_idx = {name: i for i, name in enumerate(mj_joint_names)}
            self.meta_to_mj = np.array([mj_idx[name] for name in self.meta_jnames], dtype=np.int32)

        # motion ref
        anchor_name     = md.get("anchor_body_name", None)
        body_names_meta = _csv(md.get("body_names","")) or None
        self.ref = MotionRef(motion_npz_path, n=self.n, anchor_body_name=anchor_name, body_names=body_names_meta)

        self.motion_fps = float(self.ref.data.get("fps", 30.0))
        self.ctrl_dt    = 1.0 / 60.0
        self._accum     = 0.0

        self.prev_action = np.zeros(self.n, dtype=np.float32)

        print(f"[ONNX] input='{self.inp_name}' obs_dim={self.obs_dim} output='{self.out_name}'")
        print(f"[ONNX] n_joints={self.n}, mapped_to_mj={self.meta_to_mj is not None}")
        
    # --- small helpers ---
    def reset(self):
        self.prev_action.fill(0.0)
        self.ref.t = 0
        self._accum = 0.0

    def _mj_to_meta(self, arr_mj: np.ndarray) -> np.ndarray:
        if self.meta_to_mj is None:
            return arr_mj
        return arr_mj[self.meta_to_mj]

    def _build_obs_160(self, qpos_full: np.ndarray, qvel_full: np.ndarray) -> np.ndarray:
        # base pose/vel
        base_q = qpos_full[0:4]
        base_p = qpos_full[4:7]
        R_wb   = quat_to_mat(base_q, fmt="wxyz")
        R_bw   = R_wb.T
        w_w    = qvel_full[0:3]
        v_w    = qvel_full[3:6]
        w_b    = R_bw @ w_w
        v_b    = R_bw @ v_w

        # joints (MuJoCo slices) -> metadata order
        jpos_mj  = qpos_full[FREE_BASE_QPOS:FREE_BASE_QPOS+self.n].astype(np.float32)
        jvel_mj  = qvel_full[FREE_BASE_QVEL:FREE_BASE_QVEL+self.n].astype(np.float32)
        jpos_meta = self._mj_to_meta(jpos_mj)
        jvel_meta = self._mj_to_meta(jvel_mj)

        # motion command (2n) + anchor info
        cmd_2n = np.concatenate([self.ref.jpos[self.ref.t], self.ref.jvel[self.ref.t]], dtype=np.float32)
        p_ref_w = self.ref.anchor_pos_w()
        R_ref_w = self.ref.anchor_rot_w()
        # position of motion anchor in base frame
        anchor_pos_b = R_bw @ (p_ref_w - base_p)
        # orientation error in base frame (first two columns)
        R_err_b = R_bw @ R_ref_w
        anchor_ori_6 = np.hstack([R_err_b[:,0], R_err_b[:,1]]).astype(np.float32)

        blocks = []
        for name in (self.obs_names if self.obs_names else
                     ["command","motion_anchor_pos_b","motion_anchor_ori_b",
                      "base_lin_vel","base_ang_vel","joint_pos","joint_vel","actions"]):
            if name == "command":
                blocks.append(cmd_2n)                       # 2n
            elif name == "motion_anchor_pos_b":
                blocks.append(anchor_pos_b.astype(np.float32))  # 3
            elif name == "motion_anchor_ori_b":
                blocks.append(anchor_ori_6)                 # 6
            elif name == "base_lin_vel":
                blocks.append(v_b.astype(np.float32))       # 3
            elif name == "base_ang_vel":
                blocks.append(w_b.astype(np.float32))       # 3
            elif name == "joint_pos":
                blocks.append(jpos_meta.astype(np.float32)) # n
            elif name == "joint_vel":
                blocks.append(jvel_meta.astype(np.float32)) # n
            elif name == "actions":
                blocks.append(self.prev_action.astype(np.float32)) # n
            else:
                raise RuntimeError(f"Unknown obs block: {name}")

        obs = np.concatenate(blocks, dtype=np.float32)
        if obs.size != self.obs_dim:
            sizes = [len(b) for b in blocks]
            raise RuntimeError(f"Obs dim mismatch: built {obs.size} vs model {self.obs_dim}. "
                               f"order={self.obs_names}, sizes={sizes}")
        return obs[None, :]

    def predict(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        obs = self._build_obs_160(qpos, qvel)
        act_meta = self.sess.run([self.out_name], {self.inp_name: obs})[0].astype(np.float32).ravel()

        # advance motion by ctrl dt
        self._accum += float(getattr(self, "ctrl_dt", 1.0/60.0))
        frame_dt = 1.0 / float(self.motion_fps)
        while self._accum >= frame_dt:
            self.ref.step()
            self._accum -= frame_dt

        self.prev_action = act_meta.copy()
        target_meta = self.default_joint_pos + self.action_scale * act_meta

        if self.meta_to_mj is not None:
            target_mj = np.empty_like(target_meta)
            target_mj[self.meta_to_mj] = target_meta
            return target_mj
        return target_meta

def control_loop(env, policy, output_file, save_npz=False, steps=5000):
    # --- wait for sim ---
    while not env.physics.is_ready():
        time.sleep(0.1)
    try:
        policy.ctrl_dt = float(env.physics.get_timestep())
        print(f"[Sim] dt = {policy.ctrl_dt:.5f}s")
    except Exception:
        policy.ctrl_dt = 1.0 / 60.0

    policy.reset()
    _ = env.reset()

    # --- sizes & mapping ---
    n = int(policy.n)
    meta_to_mj = getattr(policy, "meta_to_mj", None)
    if meta_to_mj is None or len(meta_to_mj) != n:
        meta_to_mj = np.arange(n, dtype=np.int32)   # identity fallback

    # --- gains/limits from ONNX (required -> fallback to sane defaults) ---
    kp_meta  = getattr(policy, "joint_stiffness", None)
    kd_meta  = getattr(policy, "joint_damping",  None)
    lim_meta = getattr(policy, "motor_effort_limits", None)

    if kp_meta is None or len(kp_meta) != n:  kp_meta  = np.full(n, 45.0, dtype=np.float32)
    if kd_meta is None or len(kd_meta) != n:  kd_meta  = np.full(n,  2.0, dtype=np.float32)
    if lim_meta is None or len(lim_meta) != n: lim_meta = np.full(n,120.0, dtype=np.float32)

    # map to MuJoCo order
    KP  = np.empty(n, dtype=np.float32);  KP[meta_to_mj]  = np.asarray(kp_meta,  dtype=np.float32)
    KD  = np.empty(n, dtype=np.float32);  KD[meta_to_mj]  = np.asarray(kd_meta,  dtype=np.float32)
    LIM = np.empty(n, dtype=np.float32); LIM[meta_to_mj] = np.asarray(lim_meta, dtype=np.float32)

    # --- set joints to ONNX default (no base teleport) ---
    try:
        st = env.physics.get_state()
        qpos0 = np.array(st["qpos"], dtype=np.float32)
        qvel0 = np.array(st["qvel"], dtype=np.float32)

        default_meta = np.asarray(getattr(policy, "default_joint_pos", np.zeros(n, np.float32)), dtype=np.float32)
        default_mj   = np.empty(n, dtype=np.float32); default_mj[meta_to_mj] = default_meta

        qpos0[7:7+n] = default_mj
        qvel0[6:6+n] = 0.0
        env.physics.set_state(qpos=qpos0.tolist(), qvel=qvel0.tolist())
        print("[init] joints -> ONNX default_joint_pos")
    except Exception as e:
        print("[init] skip set_state:", e)

    # --- logging (optional) ---
    if save_npz:
        logs = {k: [] for k in ["qpos","qvel","ctrl"]}
    else:
        logs = None

    # --- main loop: policy -> PD -> apply ---
    for step in range(steps):
        st   = env.physics.get_state()
        qpos = np.array(st["qpos"], dtype=np.float32)
        qvel = np.array(st["qvel"], dtype=np.float32)

        jpos = qpos[7:7+n]
        jvel = qvel[6:6+n]

        # policy returns target joint positions in MuJoCo order (your BMOnnxPolicy handles 160->154 inside)
        q_target = policy.predict(qpos, qvel).astype(np.float32)
        if q_target.shape[0] != n:
            q_target = q_target.reshape(-1)[:n]

        # PD (no extra base forces, no custom bias)
        tau = KP * (q_target - jpos) + KD * (-jvel)
        tau = np.clip(tau, -LIM, LIM)

        # pack controls: base (6) = 0, joints = tau
        ctrl = np.zeros(6 + n, dtype=np.float32)
        ctrl[6:] = tau

        _o, _r, done, _info = env.step(ctrl)

        if logs is not None:
            if "qpos" in st: logs["qpos"].append(np.array(st["qpos"], dtype=np.float64))
            if "qvel" in st: logs["qvel"].append(np.array(st["qvel"], dtype=np.float64))
            logs["ctrl"].append(ctrl.astype(np.float64))

        if done:
            policy.reset()
            env.reset()

    # --- save ---
    if save_npz and logs is not None:
        for k in logs: logs[k] = np.array(logs[k])
        np.savez(output_file, **logs)
        print(f"[save] NPZ -> {output_file}")
    else:
        print(f"[done] steps={steps}")


def start_server_thread(env):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    env.start()

def main():
    parser = argparse.ArgumentParser(description="Vuer MuJoCo simulation with BeyondMimic ONNX policy")
    parser.add_argument("--xml", required=True, help="Path to MJCF XML file")
    parser.add_argument("--file", required=True, help="Output file path")
    parser.add_argument("--npz", action="store_true", help="Save all keyframe data to NPZ format instead of CSV qpos only")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps to record (default: 5000)")
    args = parser.parse_args()

    # Resolve paths
    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = Path(__file__).parent / xml_path

    output_file = Path(args.file)
    if not output_file.is_absolute():
        output_file = Path(__file__).parent / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Set model + motion paths BEFORE constructing policy
    onnx_path = Path(__file__).parent / "policies" / "2025-09-25_22-13-32_g1_lafan1_dance1_s1.onnx"
    MOTION_NPZ_PATH = "/home/alecz2/motion_data/motions_lafan1_dance1_s1/motion.npz"

    mj_actuator_names_xml = get_mj_actuator_joint_names_recursive(str(xml_path))
    # drop the 6 base DoFs (they all use the same joint="floating_base_joint")
    joint_only = [j for j in mj_actuator_names_xml if j != "floating_base_joint"]
    print(f"[Model] {len(mj_actuator_names_xml)} actuators total; using {len(joint_only)} joint actuators")

    policy = BMOnnxPolicy(str(onnx_path), motion_npz_path=MOTION_NPZ_PATH,
                        mj_joint_names=joint_only)
    print("First 10 policy vs MJCF:",
        list(zip(policy.meta_jnames[:10], joint_only[:10])))

    env = BaseEnv(physics=VuerSim(mjcf_path=str(xml_path), port=8012), task=FalconTask())
    threading.Thread(target=start_server_thread, args=(env,), daemon=True).start()

    # wait for sim ready (so we can optionally query model info, if you want)
    while not env.physics.is_ready():
        time.sleep(0.1)

    # Optionally compare with names reported by the sim (if available)
    mj_actuator_names_sim = None
    try:
        info = env.physics.get_model_info()
        mj_actuator_names_sim = (
            info.get("actuator_names")
            or info.get("joint_names_actuated")
            or info.get("ctrl_names")
        )
    except Exception:
        pass

    # Choose a mapping list (must match length n). We'll pick XML list if it looks good.
    chosen_names = None
    if mj_actuator_names_xml:
        print(f"[Model] Parsed {len(mj_actuator_names_xml)} actuator joints from MJCF.")
        chosen_names = mj_actuator_names_xml
    elif mj_actuator_names_sim:
        print(f"[Model] Using actuator mapping from sim with {len(mj_actuator_names_sim)} names.")
        chosen_names = mj_actuator_names_sim

    print(f"[BMOnnxPolicy] obs_dim={policy.obs_dim}, n={policy.n}, command_dim={policy.command_dim}")
    print(f"Using XML: {xml_path}")
    print(f"Output file: {output_file}")
    print(f"Save format: {'NPZ (all keyframe data)' if args.npz else 'CSV (qpos only)'}")
    print("Open: http://localhost:8012")

    control_loop(env, policy, str(output_file), save_npz=args.npz, steps=args.steps)

if __name__ == "__main__":
    main()
