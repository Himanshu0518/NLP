// Function to toggle sections
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
}


document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    const formData = new FormData(this);

    try {
      
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

       
        const data = await response.json();

      
        document.getElementById('positive').textContent = data.positive.toFixed(2);
        document.getElementById('negative').textContent = data.negative.toFixed(2);
        document.getElementById('wordcloud').src = 'data:image/png;base64,' + data.wordcloud;
        document.getElementById('pie-chart').src = 'data:image/png;base64,' + data.pie_chart;

    } catch (error) {
       
        console.error('Error:', error);
    }
});

document.getElementById('submit-custom-text').addEventListener('click', async function() {
    const customText = document.getElementById('custom-text').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'custom-text': customText
            })
        });

        const data = await response.json();
        const positive = data.sentiment.positive;
        const negative = data.sentiment.negative;
        const neutral = data.sentiment.neutral;
        const compound = data.sentiment.compound ;

        // Display sentiment results
        document.getElementById('sentiment').textContent = `
            Positive: ${positive}, Negative: ${negative}, Neutral: ${neutral}
        `;

        // Create Pie Chart
        const pieCtx = document.getElementById('myPieChart').getContext('2d');
        new Chart(pieCtx, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    label: 'Sentiment Distribution',
                    data: [positive, negative, neutral],
                    backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe']
                }]
            },
            options: {
                responsive: false ,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });

        // Create Bar Chart
        const barCtx = document.getElementById('myBarChart').getContext('2d');
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Positive', 'Negative', 'Neutral','compound'],
                datasets: [{
                    label: 'Sentiment Score',
                    data: [positive, negative, neutral , compound],
                    backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe' , '#40ff00']
                }]
            },
            options: {
                responsive: false ,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error:', error);
    }
});
