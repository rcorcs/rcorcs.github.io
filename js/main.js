var data = {
    labels: ["2013", "2014", "2015", "2016", "2017", "2018"],
    datasets: [{
        label: "Publications",
        backgroundColor: 'rgba(54, 162, 235, 0.4)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
        data: [1, 1, 3, 5, 8, 3],
    },
    {
      label: "Citations",
      backgroundColor: 'rgba(0, 0, 0, 0.2)',
      borderColor: 'rgba(0, 0, 0, 0.4)',
      borderWidth: 1,
      data: [0, 0, 0, 3, 9, 14],
    }]
};
var options = {
    responsive: false,
    barThickness: 15,
    scales: {
        yAxes: [{
            ticks: {
                max: 15,
                min: 0,
                stepSize: 1,
            },
            stacked: false
        }],
    }
};
var ctx = document.getElementById("publicationsChart").getContext("2d");
var myBarChart = new Chart(ctx, {
    type: 'bar',
    data: data,
    options: options
});
