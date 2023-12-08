import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';


ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
  );
  
  export const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Chart.js Bar Chart',
      },
    },
  };
  
  const labels = ['AQI', 'pm25', 'pm10','O3','NO2', 'Population', 'GDP'];
  
  
  
  export function Chart(props) {
    const [firstSearch,lastSearch] = props.data
    console.log(firstSearch)
    console.log(lastSearch)
    const [data,setData] = useState( {
      labels,
      datasets: [
        {
          label: 'Dataset 1',
          data: [100,200,300,400,500,600,700],
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
        {
          label: 'Dataset 2',
          data:[100,200,300,400,500,600,700] ,
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
        },
      ],
    })
  const [firstOneData,setFirstOneData] = useState({
    "status": "success",
    "message": "data is Ok",
    "city": "Dhaka",
    "air_data": {
        "coord": {
            "lon": 90.41,
            "lat": 23.71
        },
        "list": [
            {
                "main": {
                    "aqi": 5
                },
                "components": {
                    "co": 1041.41,
                    "no": 0.09,
                    "no2": 20.39,
                    "o3": 17.88,
                    "so2": 3.31,
                    "pm2_5": 136.53,
                    "pm10": 157.05,
                    "nh3": 7.03
                },
                "dt": 1699057042
            }
        ]
    },
    "socio_data": {
        "population": 171186372,
        "populationGrowth": 1.0748367431585,
        "GDP": 460201000095.101,
        "GDPPerCapita": 2688.303950358274,
        "GDPGrowth": 7.09982877575158,
        "Country": "Bangladesh"
    }
})
  const [secondOneData,setSecondOneData] = useState({
    "status": "success",
    "message": "data is Ok",
    "city": "Dehli",
    "air_data": {
        "coord": {
            "lon": -78.52,
            "lat": -0.23
        },
        "list": [
            {
                "main": {
                    "aqi": 2
                },
                "components": {
                    "co": 1869.2,
                    "no": 56.33,
                    "no2": 43.18,
                    "o3": 0,
                    "so2": 36.72,
                    "pm2_5": 11.94,
                    "pm10": 14.64,
                    "nh3": 15.71
                },
                "dt": 1699060517
            }
        ]
    },
    "socio_data": {
        "population": 1417173173,
        "populationGrowth": 0.680372581057012,
        "GDP": 3385089881935.39,
        "GDPPerCapita": 2388.621197767614,
        "GDPGrowth": 7.00295154446648,
        "Country": "India"
    }
})
useEffect(()=>{
  if (firstSearch.isAvailable===true) {
    //console.log(areaDetatils.data);
    //const queryParams = new URLSearchParams(areaDetatils.data);
    console.log(firstSearch.data);
    const url = 'http://localhost:5000/getdata';
    
    // const fullUrl = `${url}?${queryParams.toString()}`;

axios.post(url, firstSearch.data)
  .then(response => {
    // Handle the response here
    setFirstOneData(response.data)
    console.log(response.data);
   
  })
  .catch(error => {
    // Handle errors
    console.error(error);
  });
    
  }
    },[firstSearch]);


    useEffect(()=>{
      if (lastSearch.isAvailable===true) {
        //console.log(areaDetatils.data);
        //const queryParams = new URLSearchParams(areaDetatils.data);
        console.log(lastSearch.data);
        const url = 'http://localhost:5000/getdata';
        
        // const fullUrl = `${url}?${queryParams.toString()}`;
    
    axios.post(url, lastSearch.data)
      .then(response => {
        // Handle the response here
        setSecondOneData(response.data)
        console.log(response.data);
       
      })
      .catch(error => {
        // Handle errors
        console.error(error);
      });
        
      }
        },[lastSearch]);

    useEffect(()=>{
     
        let firstOne = firstOneData.air_data.list[0].components
        let secondOne = secondOneData.air_data.list[0].components
        let firstOneSocio = firstOneData.socio_data
        let secondOneSocio = secondOneData.socio_data
        let data = {
          labels: labels,
          datasets: [
            {
              label: firstOneData.city,
              data: [firstOneData.air_data.list[0].main.aqi*10,firstOne.pm2_5,firstOne.pm10,firstOne.o3,firstOne.no2,firstOneSocio.population/10000000,firstOneSocio.GDP/10000000000],
              backgroundColor: 'rgb(255, 99, 132)',
            },
            {
              label: secondOneData.city,
              data: [secondOneData.air_data.list[0].main.aqi*10,secondOne.pm2_5,secondOne.pm10,secondOne.o3,secondOne.no2,secondOneSocio.population/10000000,secondOneSocio.GDP/10000000000],
              backgroundColor: 'rgb(54, 162, 235)',
            },
          ],
        };
        setData(data)
      
    },[firstOneData,secondOneData])
    return <Bar id='actual-chart' options={options} data={data} />;
  }
  