package sigmacorns.opmode.test

import com.qualcomm.robotcore.eventloop.opmode.TeleOp
import sigmacorns.autodog.AutoDogLink
import sigmacorns.autodog.ChaseController
import sigmacorns.autodog.FieldFrame
import sigmacorns.math.Pose2d
import sigmacorns.opmode.SigmaOpMode
import sigmacorns.subsystem.DriveController

/**
 * Auto dog: the Pi tracks balls in field coordinates (python/autodog_run.py
 * robot) and this opmode streams odometry to it and chases its target.
 *
 * Unlike VisionDriveTest, which steers on raw pixel offsets, the target here
 * is a filtered field position, so the robot keeps driving onto a ball after
 * it drops out of the camera's view just in front of the intake.
 *
 * Gamepad 1:
 *   hold A      chase the Pi's target with the intake on; with no target the
 *               robot stops. Releasing A is the dead-man switch.
 *   hold A + B  same, but spin in place to search while there is no target
 *   sticks      normal manual drive whenever A is not held
 */
@TeleOp(name = "Auto Dog Chase", group = "Test")
class AutoDogChaseTest : SigmaOpMode() {

    /** Ignore targets older than this (s): the Pi or the link has stalled. */
    private val maxTargetAge = 0.25
    private val searchTurn = 0.25

    override fun runOpMode() {
        val driveController = DriveController()
        val chase = ChaseController()
        // null host: learn the Pi's address from its first clock-sync ping
        val link = AutoDogLink(piHost = null)

        telemetry.addLine("Auto Dog ready on UDP ${link.localPort}")
        telemetry.addLine("Start the Pi: python autodog_run.py robot --rc <this RC's IP>")
        telemetry.update()

        waitForStart()
        if (isStopRequested) {
            link.close()
            return
        }

        try {
            ioLoop { state, _ ->
                val now = link.clock()
                val pose = state.driveTrainPosition
                val (rx, ry) = FieldFrame.toTracker(pose.v.x, pose.v.y)
                link.sendPose(rx, ry, pose.rot, now)

                val target = link.latestTarget
                val fresh = target != null && target.hasTarget && target.age(now) < maxTargetAge

                when {
                    gamepad1.a && fresh -> {
                        val (gx, gy) = target!!.predictedTo(now)
                        val p = chase.compute(rx, ry, pose.rot, gx, gy)
                        driveController.drive(Pose2d(p.forward, p.left, p.turn), io)
                        io.intake = 1.0
                    }
                    gamepad1.a && gamepad1.b -> {
                        driveController.drive(Pose2d(0.0, 0.0, searchTurn), io)
                        io.intake = 0.0
                    }
                    gamepad1.a -> {
                        driveController.drive(Pose2d(0.0, 0.0, 0.0), io)
                        io.intake = 0.0
                    }
                    else -> {
                        driveController.update(gamepad1, io)
                        io.intake = 0.0
                    }
                }

                telemetry.addData("Pi", if (link.pingsAnswered > 0) "synced (${link.pingsAnswered} pings)" else "waiting for ping")
                telemetry.addData("Pose (tracker frame)", "%.2f, %.2f, %.0f deg".format(rx, ry, Math.toDegrees(pose.rot)))
                if (target == null) {
                    telemetry.addData("Target", "none received")
                } else if (!target.hasTarget) {
                    telemetry.addData("Target", "Pi sees no ball")
                } else {
                    val (gx, gy) = target.predictedTo(now)
                    telemetry.addData(
                        "Target",
                        "#%d %c at %.2f, %.2f (age %.0f ms)".format(
                            target.trackId, target.color, gx, gy, 1000 * target.age(now)
                        )
                    )
                }
                telemetry.addData("Mode", if (gamepad1.a) "CHASE" else "manual")
                telemetry.update()
                false
            }
        } finally {
            driveController.drive(Pose2d(0.0, 0.0, 0.0), io)
            io.intake = 0.0
            link.close()
        }
    }
}
