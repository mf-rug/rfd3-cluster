"""rfd3_tools — generate RFD3 / MPNN Slurm jobs for HPC clusters.

Local-side: `rfd3-setup-job` builds a design JSON + a cluster-correct job.sh.
Cluster-side: this same package is shipped alongside job.sh so the job's exit
trap can parse the Slurm log into a clean summary without any cluster-side
installation (mirrors the boltz_tools arrangement).
"""

__version__ = "0.1.0"
