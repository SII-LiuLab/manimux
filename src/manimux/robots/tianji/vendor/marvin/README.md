# Marvin SDK (vendored)

Files copied unmodified from the vendor's `TJ_FX_ROBOT_CONTRL_SDK` distribution
(上海孚晞科技, Apache License 2.0, see `LICENSE`):

| File | Source | SHA-256 |
|---|---|---|
| `fx_robot.py` | `SDK_PYTHON/fx_robot.py` | `e90fbd1fb7c36c5f062fe9d06c4465e78ae117ff8354769770aeed0053fff395` |
| `fx_kine.py` | `SDK_PYTHON/fx_kine.py` | `690d26826c7b4112e3c9b088d2e21aac3154f05e6a32a368503ecd42901d048d` |
| `libMarvinSDK.so` | `SDK_PYTHON/libMarvinSDK.so` | `47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011` |
| `libKine.so` | `SDK_PYTHON/libKine.so` | `f1512f8e440b53614b8d61adc27b217cab1b97d8bb9a80e32e00c880eee73397` |
| `ccs_m6_40.MvKDCfg` | `CommonConfig/ccs_m6_40.MvKDCfg` | `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52` |
| `LICENSE` | `LICENSE` | `4f9da023eb0f5c4b776c06936fe257863620bd00a26d13225d4629b08144b1a9` |

The libraries are Linux x86-64 builds; the bindings load them from this
directory. `ccs_m6_40.MvKDCfg` is the kinematics/limits table for the CCS 6 kg
arm, controller version 4.0 (the 3.1 table differs). Load the bindings through
`manimux.robots.tianji.sdk`, which also enforces the TacCap-first import order.
This directory is excluded from lint and type checking.
