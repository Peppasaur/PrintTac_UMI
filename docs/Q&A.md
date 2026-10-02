## Q&A
### 1. Why can not my GelSight Mini get 24 FPS?
We recommend using a high-performance CPU (e.g., Core i9-13900K) and
enabling explicit CPU core binding to ensure 24 FPS for GelSight Mini.
That is because GelSight Mini has a high resolution
which requires high CPU resources and
the OS scheduler may cause delays when launching multiple processes simultaneously.
Therefore, if you are using GelSight Mini, we recommend executing the following steps to perform CPU core binding.
 1. Add config into `/etc/security/limits.conf` to ensure the user has the permission to set realtime priority.
    ```
    username - rtprio 99
    ```
 2. Edit `/etc/default/grub` and add `isolcpus=xxx` to the `GRUB_CMDLINE_LINUX_DEFAULT` line
 for isolating certain CPU cores.
 3. Modify the task configuration file and 
 the beginning several lines of all entry-point Python files (e.g. teleop.py) 
 to adjust the corresponding core binding.
