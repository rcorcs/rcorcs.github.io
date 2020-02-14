var data = {
    labels: ["2015", "2016", "2017", "2018", "2019", "2020"],
    datasets: [
    {
      label: "Citations",
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      borderColor: 'rgba(0, 0, 0, 0.7)',
      borderWidth: 1,
      data: [0, 5, 11, 27, 19, 5],
    }]
};
var options = {
    responsive: false,
    barThickness: 15,
    scales: {
        yAxes: [{
            ticks: {
                max: 30,
                min: 0,
                stepSize: 5,
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
