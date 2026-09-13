def generate_explanation(
    anomaly_type,
    confidence,
    temperature,
    humidity,
    pressure,
    data=None,
    context="Not Evaluated",
    neighbor_result=None
):

    # ==========================================
    # GENUINE WEATHER EVENT
    # ==========================================

    if context == "Likely Genuine Weather Event":

        if neighbor_result is not None:

            neighbor_avg = neighbor_result.get(
                "neighbor_temperature_avg",
                None
            )

            temperature_difference = neighbor_result.get(
                "temperature_difference",
                None
            )

            if (
                neighbor_avg is not None
                and
                temperature_difference is not None
            ):

                why = (
                    f"Temperature reached {temperature:.1f}°C "
                    f"and nearby AWS stations averaged "
                    f"{neighbor_avg:.1f}°C. The difference was only "
                    f"{temperature_difference:.2f}°C, indicating that "
                    "nearby stations experienced similar conditions."
                )

            else:

                why = (
                    "Nearby AWS stations show similar conditions, "
                    "supporting the possibility of a genuine "
                    "weather event."
                )

        else:

            why = (
                "Nearby AWS observations support the unusual "
                "weather condition."
            )

        return {
            "why": why,
            "recommendation": (
                "Treat the observation as a possible genuine "
                "weather event and continue monitoring nearby "
                "AWS stations."
            )
        }


    # ==========================================
    # LIKELY SENSOR FAULT
    # ==========================================

    if context == "Likely Sensor Fault":

        if neighbor_result is not None:

            neighbor_avg = neighbor_result.get(
                "neighbor_temperature_avg",
                None
            )

            temperature_difference = neighbor_result.get(
                "temperature_difference",
                None
            )

            if (
                neighbor_avg is not None
                and
                temperature_difference is not None
            ):

                why = (
                    f"The station reported {temperature:.1f}°C, "
                    f"while nearby AWS stations averaged "
                    f"{neighbor_avg:.1f}°C. The difference of "
                    f"{temperature_difference:.2f}°C suggests that "
                    "the local reading is not supported by nearby "
                    "stations."
                )

            else:

                why = (
                    "Nearby AWS stations do not support the "
                    "unusual observation."
                )

        else:

            why = (
                "The AWS observation appears abnormal and "
                "requires sensor verification."
            )

        return {
            "why": why,
            "recommendation": (
                "Inspect the affected sensor, verify its "
                "calibration and compare the reading with "
                "historical observations."
            )
        }


    # ==========================================
    # UNCERTAIN
    # ==========================================

    if context == "Uncertain":

        return {
            "why": (
                "The AI detected unusual AWS behavior, but "
                "there is not enough contextual evidence to "
                "determine whether it is genuine weather or "
                "a sensor problem."
            ),
            "recommendation": (
                "Compare historical observations and nearby "
                "AWS stations before taking corrective action."
            )
        }


    # ==========================================
    # NORMAL
    # ==========================================

    if anomaly_type == "Normal":

        return {
            "why": (
                "Temperature, humidity and atmospheric pressure "
                "are behaving consistently with the learned "
                "normal pattern."
            ),
            "recommendation": "No action required."
        }


    # ==========================================
    # TEMPERATURE SPIKE
    # ==========================================

    if anomaly_type == "Temperature Spike":

        max_change = None

        if (
            data is not None
            and
            "temp_change" in data.columns
        ):

            max_change = (
                data["temp_change"]
                .abs()
                .max()
            )

        if max_change is not None:

            why = (
                f"Temperature changed sharply by approximately "
                f"{max_change:.2f}°C within the recent observations."
            )

        else:

            why = (
                f"Temperature reached {temperature:.1f}°C, "
                "indicating a sudden abnormal change."
            )

        return {
            "why": why,
            "recommendation": (
                "Inspect the temperature sensor and verify "
                "its calibration."
            )
        }


    # ==========================================
    # FROZEN SENSOR
    # ==========================================

    if anomaly_type == "Frozen Sensor":

        return {
            "why": (
                "The temperature value remained unchanged "
                "across multiple observations while other "
                "weather variables changed."
            ),
            "recommendation": (
                "Check whether the temperature sensor is "
                "stuck or malfunctioning."
            )
        }


    # ==========================================
    # SENSOR DRIFT
    # ==========================================

    if anomaly_type == "Sensor Drift":

        return {
            "why": (
                "Temperature shows a gradual and persistent "
                "change over multiple observations, which "
                "may indicate sensor drift."
            ),
            "recommendation": (
                "Inspect the sensor and perform calibration "
                "if necessary."
            )
        }


    # ==========================================
    # COMMUNICATION ERROR
    # ==========================================

    if anomaly_type == "Communication Error":

        return {
            "why": (
                "The AWS transmission contains missing or "
                "invalid readings, indicating a possible "
                "communication or data logger problem."
            ),
            "recommendation": (
                "Check the AWS communication link, data "
                "logger and network connection."
            )
        }


    # ==========================================
    # MULTIVARIATE INCONSISTENCY
    # ==========================================

    if anomaly_type == "Multivariate Inconsistency":

        abnormal_details = []

        if data is not None:

            if "humidity" in data.columns:

                invalid_humidity = data[
                    (data["humidity"] < 0) |
                    (data["humidity"] > 100)
                ]

                if not invalid_humidity.empty:

                    max_humidity = (
                        invalid_humidity["humidity"].max()
                    )

                    abnormal_details.append(
                        f"humidity reached {max_humidity:.1f}%"
                    )

            if "pressure" in data.columns:

                invalid_pressure = data[
                    (data["pressure"] < 850) |
                    (data["pressure"] > 1100)
                ]

                if not invalid_pressure.empty:

                    max_pressure = (
                        invalid_pressure["pressure"].max()
                    )

                    abnormal_details.append(
                        f"pressure reached {max_pressure:.1f} hPa"
                    )

        if abnormal_details:

            why = (
                "An abnormal observation was detected within "
                "the recent AWS sequence. "
                + " and ".join(abnormal_details)
                + ", which are outside the expected "
                "atmospheric ranges."
            )

        else:

            why = (
                f"The combination of temperature "
                f"({temperature:.1f}°C), humidity "
                f"({humidity:.1f}%) and pressure "
                f"({pressure:.1f} hPa) appears inconsistent "
                "with expected atmospheric conditions."
            )

        return {
            "why": why,
            "recommendation": (
                "Verify the affected sensor readings and "
                "compare them with nearby AWS observations "
                "before accepting the data."
            )
        }


    # ==========================================
    # FALLBACK
    # ==========================================

    return {
        "why": (
            "The system detected unusual AWS behavior "
            "that requires review."
        ),
        "recommendation": (
            "Perform manual verification of the AWS readings."
        )
    }