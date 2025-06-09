from ..base_loss import baseLoss
import numpy as np


# Table 11.6 from Kakaç, Sadık. "Heat Exchangers: Selection, Rating, and Thermal Design." Heat Exchangers, n.d.
pressure_loss_data = {
    30: {
        0:   {"Kp": 50.000, "m": 1.000},
        10:  {"Kp": 19.400, "m": 0.589},
        100: {"Kp": 2.990,  "m": 0.183},
    },
    45: {
        0:   {"Kp": 47.000, "m": 1.000},
        10:  {"Kp": 18.290, "m": 0.652},
        100: {"Kp": 1.441,  "m": 0.206},
    },
    50: {
        0:   {"Kp": 34.000, "m": 1.000},
        20:  {"Kp": 11.250, "m": 0.631},
        300: {"Kp": 0.772,  "m": 0.161},
    },
    60: {
        0:   {"Kp": 24.000, "m": 1.000},
        20:  {"Kp": 3.240,  "m": 0.457},
        400: {"Kp": 0.760,  "m": 0.215},
    },
    65: {
        0:   {"Kp": 24.000, "m": 1.000},
        20:  {"Kp": 2.800,  "m": 0.451},
        500: {"Kp": 0.639,  "m": 0.213},
    },
}


class ChannelLoss(baseLoss):
    """Pressure drop model from Kakaç, Sadık. "Heat Exchangers: Selection, Rating, and Thermal Design." Heat Exchangers, n.d."""

    def get_friction_factors(self, Re):
        """Get the friction factors Kp and m based on chevron angle and Reynolds number.

        Args:
            Re (float): Reynolds number

        Returns:
            tuple[float, float]: Kp and m values for the given conditions
        """
        # Find the closest chevron angle in the data
        available_angles = sorted(pressure_loss_data.keys())
        chevron_angle = self.geometry.chevron_angle

        if chevron_angle <= available_angles[0]:
            angle = available_angles[0]
        elif chevron_angle >= available_angles[-1]:
            angle = available_angles[-1]
        else:
            # Find the closest angle
            angle = min(available_angles, key=lambda x: abs(x - chevron_angle))

        # Get the Reynolds number ranges for the selected angle
        Re_ranges = sorted(pressure_loss_data[angle].keys())

        # Find the appropriate Reynolds number range
        if Re <= Re_ranges[0]:
            Re_key = Re_ranges[0]
        elif Re >= Re_ranges[-1]:
            Re_key = Re_ranges[-1]
        else:
            # Find the closest Reynolds number range
            Re_key = min(Re_ranges, key=lambda x: abs(x - Re))

        # Get the friction factors
        factors = pressure_loss_data[angle][Re_key]
        return factors["Kp"], factors["m"]

    def evaluate(self, thermo, w):
        """Calculate pressure loss in the plate heat exchanger channel.

        Args:
            thermo: Thermodynamic state object
            w (float): Mass flow rate

        Returns:
            float: Pressure loss in Pa
        """
        # Get mass flux per channel
        G_channel = w/self.geometry._area_channel_total

        # Calculate Reynolds number
        Re = G_channel * self.geometry._Dh / thermo._viscosity

        # Get friction factors
        Kp, m = self.get_friction_factors(Re)

        # Calculate friction factor
        f = Kp/Re**m

        # Calculate channel pressure drop
        delta_p_channel = (
            f * (self.geometry._Lp / self.geometry._Dh) * thermo._density
            * (G_channel**2) / 2
        )

        # Port mass velocity
        G_p = w / (np.pi * (self.geometry._D_port**2) / 4)

        # Calculate port pressure drop
        delta_p_port = 1.4 * thermo._density * (G_p**2) / 2

        # Calculate total pressure drop
        dP = delta_p_channel + delta_p_port

        return dP
