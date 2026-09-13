def validate_with_neighbors(
    station_temperature,
    station_humidity,
    station_pressure,
    neighbor_data
):
    """
    Compare the current AWS station with nearby AWS stations.

    neighbor_data example:

    [
        {
            "temperature": 44.5,
            "humidity": 61,
            "pressure": 1001
        },
        {
            "temperature": 45.2,
            "humidity": 59,
            "pressure": 1000
        }
    ]
    """

    # ==========================================
    # 1. CHECK NEIGHBOR DATA
    # ==========================================

    if not neighbor_data:

        return {
            "neighbor_status": "NO_DATA",
            "weather_support": False,
            "message": (
                "No nearby AWS station data is available "
                "for comparison."
            )
        }


    # ==========================================
    # 2. CALCULATE NEIGHBOR AVERAGES
    # ==========================================

    neighbor_temperature_avg = sum(
        station["temperature"]
        for station in neighbor_data
    ) / len(neighbor_data)

    neighbor_humidity_avg = sum(
        station["humidity"]
        for station in neighbor_data
    ) / len(neighbor_data)

    neighbor_pressure_avg = sum(
        station["pressure"]
        for station in neighbor_data
    ) / len(neighbor_data)


    # ==========================================
    # 3. CALCULATE DIFFERENCES
    # ==========================================

    temperature_difference = abs(
        station_temperature -
        neighbor_temperature_avg
    )

    humidity_difference = abs(
        station_humidity -
        neighbor_humidity_avg
    )

    pressure_difference = abs(
        station_pressure -
        neighbor_pressure_avg
    )


    # ==========================================
    # 4. CHECK EACH VARIABLE
    # ==========================================

    temperature_consistent = (
        temperature_difference <= 5
    )

    humidity_consistent = (
        humidity_difference <= 15
    )

    pressure_consistent = (
        pressure_difference <= 10
    )


    # ==========================================
    # 5. WEATHER SUPPORT DECISION
    # ==========================================

    # Temperature is the most important variable
    # when checking whether a temperature anomaly
    # is a genuine regional weather event.

    if (
        temperature_consistent
        and
        humidity_consistent
        and
        pressure_consistent
    ):

        neighbor_status = "SUPPORTED"
        weather_support = True

        message = (
            "Nearby AWS stations show similar temperature, "
            "humidity and pressure conditions. The unusual "
            "reading may represent a genuine weather event."
        )

    else:

        neighbor_status = "NOT_SUPPORTED"
        weather_support = False

        message = (
            "Nearby AWS stations do not show sufficiently "
            "similar conditions. The reading may indicate "
            "a station or sensor problem."
        )


    # ==========================================
    # 6. RETURN RESULT
    # ==========================================

    return {

        "neighbor_status":
            neighbor_status,

        "weather_support":
            weather_support,

        "neighbor_temperature_avg":
            round(neighbor_temperature_avg, 2),

        "neighbor_humidity_avg":
            round(neighbor_humidity_avg, 2),

        "neighbor_pressure_avg":
            round(neighbor_pressure_avg, 2),

        "temperature_difference":
            round(temperature_difference, 2),

        "humidity_difference":
            round(humidity_difference, 2),

        "pressure_difference":
            round(pressure_difference, 2),

        "message":
            message
    }