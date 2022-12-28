var data = {
    labels: ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"],
    datasets: [
    {
      label: "Citations",
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      borderColor: 'rgba(0, 0, 0, 0.7)',
      borderWidth: 1,
      data: [0, 5, 11, 28, 19, 30,  64, 77],
    }]
};
var options = {
    responsive: false,
    barThickness: 10,
    scales: {
        yAxes: [{
            ticks: {
                max: 80,
                min: 0,
                stepSize: 10,
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
