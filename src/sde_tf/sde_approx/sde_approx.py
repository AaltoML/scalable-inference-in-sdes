"""Interface for SDE approximations."""


class SDEApprox:

    def __init__(self):
        pass

    def get_nxt_data(self, t, x):
        """Evaluate the moment ODEs at (t, x)."""
        raise NotImplementedError('ODE step not implemented')
