.PHONY: default ball_sorting mug_tree tie_knot weighted_cubes

default:
	echo "\n\033[91mPlease specify the environment you are looking for -->\033[0m"
mug_tree:
	-doc-gen --dg.env-name MugTree-fixed-v1 &
	-doc-gen --dg.env-name MugTree-mug_rand-v1 & wait
obj_permanence:
	-doc-gen --dg.env-name ObjPermanence-fixed-v1 &
	-doc-gen --dg.env-name ObjPermanence-ball_rand-v1 & wait
insert_shapes:
	-doc-gen --dg.env-name InsertShapes-fixed-v1 &
	-doc-gen --dg.env-name InsertShapes-shapes_rand-v1 &
	-doc-gen --dg.env-name InsertShapes-box_rand-v1 & wait
fishing_toy:
	-doc-gen --dg.env-name FishingToy-fixed-v1 --dg.image=front/rgb,right/rgb & wait
kitchen_room:
	-doc-gen --dg.env-name KitchenRoom-v1
tie_knot:
	-doc-gen --dg.env-name TieKnot-v1
weighted_cubes:
	-doc-gen --dg.env-name WeightedCubes-v1
