// Function to update sensor readings
function updateSensorData() {
    fetch('/api/sensor-data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('current').textContent = data.current;
            document.getElementById('voltage').textContent = data.voltage;
        });
}

// Function to update machine learning predictions
function updateMLPredictions() {
    // Get current sensor data
    const voltage = parseFloat(document.getElementById('voltage').textContent);
    const current = parseFloat(document.getElementById('current').textContent);

    // Send sensor data to the backend for prediction
    fetch('/api/ml-predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voltage: voltage, current: current })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            // Update predicted energy usage
            document.getElementById('ml-prediction').textContent = data.prediction.toFixed(2);

            // Update total price in LKR
            document.getElementById('total-price-lkr').textContent = data.total_price_lkr.toFixed(2);

            // Update the hourly power consumption graph
            updateHourlyPowerConsumptionGraph(data.hourly_data); // Assuming the backend sends hourly data
        } else {
            console.error("Failed to fetch ML predictions:", data.message);
        }
    });
}

// Function to update the hourly power consumption graph
function updateHourlyPowerConsumptionGraph(hourlyData) {
    const ctx = document.getElementById('hourly-power-consumption').getContext('2d');
    if (window.hourlyChart) {
        window.hourlyChart.destroy(); // Destroy existing chart if it exists
    }
    window.hourlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hourlyData.labels, // Array of hours (e.g., ["00:00", "01:00", ...])
            datasets: [{
                label: 'Hourly Power Consumption (kWh)',
                data: hourlyData.values, // Array of power consumption values
                borderColor: '#1abc9c',
                fill: false,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Power Consumption (kWh)'
                    }
                }
            }
        }
    });
}

// Update sensor data and ML predictions every 2 seconds
setInterval(() => {
    updateSensorData();
    updateMLPredictions();
}, 2000);

// Appliance control buttons
const lightsSwitch = document.getElementById('lights-switch');
const lightsStatus = document.getElementById('lights-status');
let lightsOn = false;

lightsSwitch.addEventListener('click', async () => {
    lightsOn = !lightsOn;
    lightsStatus.textContent = lightsOn ? 'On' : 'Off';
    lightsSwitch.classList.toggle('on', lightsOn);

    await fetch('/api/lights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: lightsOn ? 'on' : 'off' })
    });
    alert(`Lights turned ${lightsOn ? 'on' : 'off'}`);
});

document.getElementById('fan-on').addEventListener('click', async () => {
    await fetch('/api/fan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'on' })
    });
    alert('Fan turned on');
});

document.getElementById('fan-off').addEventListener('click', async () => {
    await fetch('/api/fan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'off' })
    });
    alert('Fan turned off');
});

document.getElementById('fridge-on').addEventListener('click', async () => {
    await fetch('/api/fridge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'on' })
    });
    alert('Fridge turned on');
});

document.getElementById('fridge-off').addEventListener('click', async () => {
    await fetch('/api/fridge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'off' })
    });
    alert('Fridge turned off');
});

document.getElementById('fridge-hibernate').addEventListener('click', async () => {
    await fetch('/api/fridge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'hibernate' })
    });
    alert('Fridge in hibernate mode');
});

document.getElementById('night-mode').addEventListener('click', async () => {
    await fetch('/api/night-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'toggle' })
    });
    alert('Night mode toggled');
});