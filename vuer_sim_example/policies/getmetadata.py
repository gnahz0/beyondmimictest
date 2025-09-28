PATH = "/home/alecz2/whole_body_tracking/logs/rsl_rl/g1_flat/2025-09-25_22-13-32_g1_lafan1_dance1_s1/2025-09-25_22-13-32_g1_lafan1_dance1_s1.onnx"
import onnx, json

m = onnx.load(PATH)
md = {p.key: p.value for p in m.metadata_props}

print("Metadata keys:", list(md.keys()))
# These two are the most useful for mapping the obs vector:
print("observation_names:", md.get("observation_names"))
print("command_names:", md.get("command_names"))