# Station templates

Use the [station guide](../../../docs/usage/station.md) for device bindings,
service addresses, checkpoint paths and configuration inspection.

- `yam_example.yaml`: YAM with the configured camera services.
- `tianji_taccap_example.yaml`: Tianji controller and TacCap components.
- `piper_example.yaml`: standard PiPER CAN and measured width calibration.
- `arx_x5_example.yaml`: X5 (2023) CAN, separate encoder/command calibration and
  explicit SDK polling selection. See [SDK adapter limits](../../../docs/usage/can-arms.md).
- Copy the matching template to `station.yaml`, the default private station file.

Template names use underscores; retain the `.yaml` extension. Additional private
stations can live under the ignored root `.local/` directory, for example
`.local/station_lab.yaml`, selected with `--local` on each relevant process.
