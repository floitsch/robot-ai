# What real cheap arms accept, and what related work does

A bounded literature and product search from 2026-09-24. It set how our simulated arms are driven. Every
link below was opened at the time; claims we could not confirm are marked *unverified*.

## Findings

- **Cheap arms take position targets, not torque.** Hobby servos, the Feetech STS3215 (SO-100/SO-101),
  Dynamixel XL430 and closed-loop steppers all do. Their feedback loop runs inside the servo, at about
  1 kHz on its own encoder. Of the devices checked, only the Dynamixel XL330 (current mode), moteus and
  ODrive accept torque. moteus and ODrive also accept a position target with feedforward torque and
  per-command gain scaling.
- **So the network should move servo targets.** Host latency and host-side encoder defects then only
  delay how fast targets change. The servo's own loop, which holds the arm, never sees them. With a 100 Hz
  torque loop, the worst arms' roughly 100 ms of combined delay forced every controller to be slow,
  including one with perfect knowledge.
- **Related work agrees:**
  - Peng & van de Panne (2017): PD-target actions learn fastest, are among the most robust, and degrade
    least at low control rates. Torque actions degrade fastest.
  - Hwangbo et al. (2019), Rudin et al. (2021), RMA (2021) and ALOHA (2023) all send position targets to
    a PD or PID loop.
  - Tan et al. (2018): without an actuator model and simulated latency, policies failed on the real robot.
- **Buzzing is real on these arms too.** LeRobot halves the STS3215's proportional gain (32 to 16)
  "to avoid shakiness". Differencing a 12-bit encoder at 100 Hz turns a single count into a speed step
  of 0.15 rad/s.
- **Is 5 mm realistic?**
  - Published repeatability: myCobot 280 ±0.5 mm, UFACTORY Lite 6 0.5 mm, OpenManipulator-X <0.2 mm,
    uArm Swift Pro (stepper) 0.2 mm, uArm Swift (RC servo) 5 mm.
  - One measured SO-101: 6.9 mm RMS forward-kinematics error from commanded angles, 1.8 mm from the
    angles the servos report. Reaching new targets missed by 20 mm on average, and by 5.2 mm after camera
    calibration.
  - 5 mm judged on joint encoders is a fair simulation bar. Absolute accuracy on real SO-class arms also
    needs calibration or vision.

## Real interfaces

| Device | Commands | Inner loop |
| --- | --- | --- |
| Hobby RC servo | position (pulse width, about every 20 ms) | rate *unverified* |
| Feetech STS3215 | position, closed-loop speed, open-loop PWM, step; torque limit and acceleration registers; current readable; 12-bit magnetic encoder | rate *unverified* |
| Dynamixel XL430 | velocity, position, extended position, PWM | about 1 kHz (vendor forum, exact rate not disclosed) |
| Dynamixel XL330 | current, velocity, position, current-limited position, PWM | as XL430 |
| MKS SERVO42D/57D | speed and position over pulse, RS485 or CAN; configurable PID | torque mode and rate *unverified* |
| ODrive | cascaded position, velocity and current loops; position, velocity and torque inputs with feedforward | 8 kHz |
| moteus | torque = integrator + kp·kp_scale·position error + kd·kd_scale·velocity error + feedforward | 30 kHz |

## What we adopted

- **Servo model.** Every physics step (1 ms), the servo computes a PWM duty from its target error and
  filtered velocity, reading its own undelayed 12-bit encoder. A deadband, back-EMF and a current limit
  follow. Gains vary per joint over 4×.
- **Policy output.** The policy moves each target by up to ±0.3 rad around the inverse-kinematics
  solution. "Do nothing" is then exactly what LeRobot users do, and that is the baseline.
- **Kept from before.**
  - The existing host-side defects: latency and encoder errors.
  - Teacher → student distillation (Lee et al. 2020).
  - Recurrent adaptation from proprioception (as in RMA).
  - A critic warm-up for residual learning (Silver et al.).

## Candidates for later

- An actuator network fitted to logged servo data once real hardware is available (Hwangbo et al. 2019).
- Smoothness regularization: CAPS (Mysore et al.), or low-pass filtered actions (Peng et al. 2020).
- An auxiliary loss making the network's memory predict the robot's condition (RMA).
- Optional stiffness-scale and feedforward-torque outputs for drivers that support them (moteus, ODrive,
  XL330). Variable impedance helps in contact-rich tasks (VICES, Varin et al., Bogdanovic et al.).

## Obstacles (for later)

Sensorless collision detection uses a generalized-momentum observer (Haddadin, De Luca & Albu-Schäffer
2017). Its residual approximates the external joint torque from joint positions, velocities and motor
current alone; friction modelling decides the false-alarm rate. For reacting to a hit, De Luca et al.
(2006) argue against a plain stop, which can pin a person, and retreat along the direction of the
residual instead.

## References

- ROBOTIS XL330: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- ROBOTIS XL430: https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
- Feetech STS3215: https://www.feetechrc.com/2020-05-13_56655.html
- LeRobot Feetech driver: https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py
- LeRobot SO-100 "avoid shakiness": https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/robots/so100_follower/so100_follower.py
- MKS SERVO42D/57D: https://github.com/makerbase-motor/MKS-SERVO42D-57D
- ODrive control: https://docs.odriverobotics.com/v/0.5.5/control.html
- moteus theory: https://mjbots.github.io/moteus/reference/theory/
- Peng & van de Panne 2017: https://arxiv.org/abs/1611.01055
- Rudin et al. 2021: https://arxiv.org/abs/2109.11978
- Hwangbo et al. 2019: https://arxiv.org/abs/1901.08652
- Lee et al. 2020: https://arxiv.org/abs/2010.11251
- Kumar et al. 2021 (RMA): https://arxiv.org/abs/2107.04034
- Peng et al. 2018: https://arxiv.org/abs/1710.06537
- Tan et al. 2018: https://arxiv.org/abs/1804.10332
- Johannink et al. 2019: https://arxiv.org/abs/1812.03201
- Silver et al.: https://arxiv.org/abs/1812.06298
- Martín-Martín et al. (VICES): https://arxiv.org/abs/1906.08880
- Mysore et al. (CAPS): https://arxiv.org/abs/2012.06644
- Peng et al. 2020: https://arxiv.org/abs/2004.00784
- Zhao et al. 2023 (ALOHA): https://arxiv.org/abs/2304.13705
- myCobot 280: https://www.elephantrobotics.com/en/mycobot-280-pi-2023-specifications/
- UFACTORY Lite 6: https://www.ufactory.us/product/lite-6
- OpenManipulator-X: https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/
- uArm Swift: http://download.ufactory.cc/docs/en/uArm-Swift-Specifications-171012.pdf
- SO-101 measurement (vbody): https://github.com/vladmarian20005/vbody
- Haddadin et al. 2017: http://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf
- De Luca et al. 2006: https://www.diag.uniroma1.it/deluca/pHRI_elective/IROS06_ADL_etal_CollisionDetectionReaction_DLRIII.pdf
