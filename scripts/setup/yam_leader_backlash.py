"""Independent readout for motorized YAM leaders; never starts teleoperation."""

if __package__:
    from scripts.setup.yam_backlash import main
else:
    from yam_backlash import main


if __name__ == "__main__":
    main(arm_role="leader")
