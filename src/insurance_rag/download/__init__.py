"""Download scripts for insurance data sources.

Medicare-specific downloaders (IOM, MCD, codes) are kept here for backward
compatibility.  New domains register their downloaders via the domain plugin
interface and are dispatched through the CLI ``--domain`` flag.
"""
from insurance_rag.download.codes import download_codes
from insurance_rag.download.iom import download_iom
from insurance_rag.download.mcd import download_mcd

__all__ = ["download_iom", "download_mcd", "download_codes"]
