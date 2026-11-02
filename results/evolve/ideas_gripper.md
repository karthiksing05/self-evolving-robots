# Ideas System 2 recorded (the outer loop's input)

## shoulder gravity-compensation spring (1 times, 1 runs)
- **shoulder gravity-compensation spring** (gripper_s2_s0, iteration 2): A printed bracket on the base and a printed anchor on the upper arm hold an extension spring (or a stack of 2-3 springs) across the shoulder-lift joint, offset from the axis so its moment roughly follows the arm's cos(theta) gravity moment. It offsets about 0.3-0.4 N*m of the 0.41 N*m the arm needs to hold itself, which frees servo torque for payload (0.15 N*m per 100 g). Elbow and wrist servos ar
  - addresses: arm_stall on tuna_fish_can (171 g), pudding_box (187 g) and mustard_bottle (431 g); sugar_box at 514 g will hit it next. An arm joint reaches its 1.62 N*m stall; open then: squeezed_down_out x9, slipped_under_load x6, arm_stall x4, moving_jaw_on_top_face x3
  - needs: extension spring set, about 1 N/mm, 40 mm free length (new item, est. $6, 3 d), m3_screw x4 (on hand), print:PLA 20g (about 100 machine minutes)

## long-stroke prismatic jaw (1 times, 1 runs)
- **long-stroke prismatic jaw** (gripper_s2_s0, iteration 4): A rack and pinion on two 3x150 mm ground steel rods with a 75 mm/rad pitch-radius pinion slides the moving finger straight along the closing axis over about 103 mm. The face stays parallel and the fingertip stays at the fixed fingertip's height at every opening, so it reaches below the top of 34-58 mm tall objects up to 96 mm wide.
  - addresses: The arm_stall and moving_jaw_on_top_face results on the tuna can (85 mm), pudding box (90), sugar box (92) and mustard bottle (96) are one problem. At 85 mm the; open then: squeezed_down_out x9, slipped_under_load x6, arm_stall x4, moving_jaw_on_top_face x3
  - needs: steel_rod_3x150 x2 (new item, est. $2 each, 2 d), lm3uu_bearing x2, m3_screw x6, print:PLA 40g

## crowned (convex-cylinder) moving pad (1 times, 1 runs)
- **crowned (convex-cylinder) moving pad** (gripper_s2_s0, iteration 7): A printed moving-jaw pad whose face is a convex circular arc of about 35 mm radius in the vertical plane, spanning about ±35 deg and about 30 mm tall, with a 3 mm TPU skin (mu 0.9) and the existing 5 mm V-groove running along it. Against a flat vertical side, contact always happens where the arc's tangent is vertical, so the squeeze stays horizontal at any hinge angle. The jaw's turning only moves
  - addresses: squeezed_down_out on flat-sided and large objects (gelatin box: 33 deg down at 0.88 rad; apple 26 deg; soup can 27 deg; tennis ball 20 deg). On the hinge, a fla; open then: squeezed_down_out x9, slipped_under_load x5, moving_jaw_on_top_face x4, arm_stall x3
  - needs: m2_screw x4 (on hand), print:PLA 4g (20 machine min), print:TPU 2g (16 machine min)

## self-aligning pivoting moving pad (1 times, 1 runs)
- **self-aligning pivoting moving pad** (gripper_s2_s0, iteration 9): The moving pad is mounted on a horizontal pin through its mid-height, parallel to the jaw hinge, running in two MR63 bearings. A light torsion spring centres it about 20 deg back, and it can rotate ±30 deg. On contact the pad rotates under the contact moment until its face lies flat against the object's side, so the normal is horizontal and opposed to the fixed face at every opening from 17 to 85 
  - addresses: squeezed_down_out on gelatin_box, apple and tennis_ball. At 0.69-0.86 rad the moving face is 15-27 deg bottom-out (brief: 22 deg at 65 mm, 30 deg at 75 mm). The; open then: squeezed_down_out x6, slipped_under_load x6, moving_jaw_on_top_face x4, arm_stall x3
  - needs: mr63_bearing x2 (buy $0.9 each, 3 d), m3_screw x1 (on hand, as the pivot pin), m2_screw x4 (on hand), small torsion spring about 0.02 N*m/rad (new item, est. $2, 3 d), print:PLA 4g, print:TPU 1g

## full-height vertically concave TPU pad (1 times, 1 runs)
- **full-height vertically concave TPU pad** (gripper_s2v2_s0, iteration 7): A printed PLA pad that covers the full height of each jaw face, with a TPU skin. Its face profile in the vertical plane is a shallow V or arc: two contact bands about 12 mm above and below the pad centre, about 2 mm sagitta. A lying cylinder or oval (mustard, soup can) or a sphere (apple) touches both bands. Their normals tilt up and down by equal angles, so the net vertical push cancels and the b
  - addresses: squeezed_down_out on the lying mustard: the moving pad's bottom edge meets the oval at +8 mm, above its widest point, so the contact normal tilts 15 deg down. s; open then: squeezed_down_out x2, slipped_under_load x1
  - needs: m2_screw x4 (on hand), print:PLA ~3 g (~15 min), print:TPU ~2 g (~16 min)

