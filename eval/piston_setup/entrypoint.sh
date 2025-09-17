#!/bin/sh
set -e

echo '--- Starting Gold Standard Entrypoint ---'

# 1. Create a proper home directory for the piston user
echo 'Creating home directory...'
mkdir -p /home/piston
chown -R piston:piston /home/piston

# 2. Create data directories on the REAL path
echo 'Creating data directories...'
mkdir -p /piston_api/packages

# 3. Fix all ownership and permissions on the REAL path
echo 'Fixing permissions...'
chown -R piston:piston /piston_api

# 4. Create the compatibility symlink AFTER the target is fully prepared
echo 'Creating compatibility symlink...'
ln -s /piston_api /piston

# 5. Configure cgroups (known good)
echo 'Configuring cgroups...'
cd /sys/fs/cgroup
mkdir -p isolate/init
echo 1 > isolate/cgroup.procs
echo '+cpuset +cpu +io +memory +pids' > cgroup.subtree_control
cd isolate
mkdir -p init
echo 1 > init/cgroup.procs
echo '+cpuset +memory' > cgroup.subtree_control
echo 'Cgroups configured.'

# 6. Launch the server, setting the HOME variable explicitly
echo 'Starting Piston server...'
exec su -p -- piston -c 'export HOME=/home/piston && ulimit -n 65536 && cd /piston_api && node src/index.js'

