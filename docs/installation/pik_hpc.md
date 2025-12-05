# HPC setup at PIK


## Gurobi

> [!NOTE] Set-up on the PIK cluster
>    Gurobi license activation from the compute nodes requires internet access. The workaround is an ssh tunnel to the login nodes, which can be set-up on the compute nodes with
```bash
    # interactive session on the compute nodes
    srun --qos=priority --pty bash
    # key pair gen (here ed25518 but can be rsa)
    ssh-keygen -t ed25519 -f ~/.ssh/id_rsa.cluster_internal_exchange -C "$USER@cluster_internal_exchange"
    # leave the compute nodes
    exit
```

> You will then need to add the contents of the public key `~/.ssh/id_rsa.cluster_internal_exchange.pub` to your authorised `~/.ssh/authorized_keys`, eg. with `cat <key_name> >> authorized_keys`

> TROUBLE SHOOTING
> you may have some issues with the solver tunnel failing (permission denied). One of these two steps should solve it
> option 1: name the exchange key `id_rsa`.
> option 2: copy the contents to authorized_keys from the compute nodes (from the ssh_dir `srun --qos=priority --pty bash; cat <key_name> >> authorized_keys;exit`)

> In addition you should have your .profile & .bashrc setup as per https://gitlab.pik-potsdam.de/rse/rsewiki/-/wikis/Cluster-Access
and add `module load anaconda/2024.10` (or latest) to it

## HPC profile

The model comes with a pre-defined slurm profile that has been tailored to the PIK HPC. It can be found in `configs/pik_hpc_profile/` 