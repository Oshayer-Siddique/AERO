import React, { useState } from "react";
import { useContext, useEffect } from "react";
import { Context } from "../App";
import axios from "axios";
import GoogleAnother from "../Components/GoogleAnother";
import SearchComponent from "../Components/SearchComponent";
const User = () => {
  const  [userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo] =
    useContext(Context);
const colorCode ={
  1:"bg-green-400",
  2:"bg-yellow-400",
  3:"bg-orange-400",
  4:"bg-red-400",
  5:"bg-purple-400"

}


const [color,setColor] = useState(cityFullInfo.air_data.list[0].main.aqi)
console.log(colorCode[color])
    const [tempData,setTempData] = useState(
     {
        isAvailable:false,
        address:{
          "isAvailable": true,
         
              "city": "P",
              "state": "Dhaka",
              "country": "Bangladesh",
              "location": {
                  "type": "Point",
                  "coordinates": [
                      90.41426,
                      23.73625
                  ]
              },
              "current": {
                "pollution": {
                    "ts": "2023-11-03T09:00:00.000Z",
                    "aqius": 119,
                    "mainus": "p2",
                    "aqicn": 60,
                    "maincn": "p2"
                },
               
            
          }
      },
      })

useEffect(()=>{
  if (areaDetatils.isAvailable===true) {
    //console.log(areaDetatils.data);
    //const queryParams = new URLSearchParams(areaDetatils.data);
    console.log(areaDetatils.data);
    const url = 'http://34.42.153.40:5000/getdata';
    
    // const fullUrl = `${url}?${queryParams.toString()}`;

axios.post(url, areaDetatils.data)
  .then(response => {
    // Handle the response here
    setCityFullInfo(response.data)
    console.log(response.data);
    setColor(response.data.air_data.list[0].main.aqi)
    
  })
  .catch(error => {
    // Handle errors
    console.error(error);
  });
    
  }
    },[areaDetatils]);

    useEffect(()=>{
      setTempData(
        {
          isAvailable:true,
          address:{
            
                "city": cityFullInfo.city,
                "state": "Dhaka",
                "country": cityFullInfo.socio_data.Country,
                "location": {
                    "type": "Point",
                    "coordinates": [
                      cityFullInfo.air_data.coord.lon,
                      cityFullInfo.air_data.coord.lat
                    ]
                },
                "current": {
                  "pollution": {
                      "ts": "2023-11-03T09:00:00.000Z",
                      "aqius": cityFullInfo.air_data.list[0].main.aqi,
                      "mainus": "p2",
                      "aqicn": 60,
                      "maincn": "p2"
                  },
                 
              }
            
        },
        }
  
      )
      
    },[cityFullInfo]);
  return (
    <div id="user-page" className="flex gap-2 flex-col md:flex-row">
      <div className="border-2 rounded-md">
       <div id="details-box" className="w-[30vw] p-28 text-xl ">
       <h1>{cityFullInfo.city}</h1>
       <h2>Country: {cityFullInfo.socio_data.Country}</h2>
        <h1>Status</h1>
        <div>
          <h2 className="border-b-2">AQI : {cityFullInfo.air_data.list[0].main.aqi}</h2>
          <h2 className="border-b-2">CO : {cityFullInfo.air_data.list[0].components.co}</h2>
          <h2 className="border-b-2">NO2 : {cityFullInfo.air_data.list[0].components.no2}</h2>
          <h2 className="border-b-2">O3 : {cityFullInfo.air_data.list[0].components.o3}</h2>
          <h2 className="border-b-2">pm10 : {cityFullInfo.air_data.list[0].components.pm10}</h2>
          <h2 className="border-b-2">pm25 : {cityFullInfo.air_data.list[0].components.pm25}</h2>
  
          <h2 className="border-b-2">so2 : {cityFullInfo.air_data.list[0].components.so2}</h2>
          <h2 className="border-b-2">population: {cityFullInfo.socio_data.population}</h2>
          <h2 className="border-b-2">populationGrowth: {cityFullInfo.socio_data.populationGrowth}</h2>
          <h2 className="border-b-2">GDP: {cityFullInfo.socio_data.GDP}</h2>
          <h2 className="border-b-2">GDPPerCapita: {cityFullInfo.socio_data.GDPPerCapita}</h2>
          <h2 className="border-b-2">GDPGrowth: {cityFullInfo.socio_data.GDPGrowth}</h2>
         


        </div>
       </div>
      </div>
      <div id="right-side-user" className="w-[60vw]">
        <div id="bar-in-user" className={`${colorCode[color]}!important`}><h1 className="text-2xl">{cityFullInfo.city}</h1>
        <h1 className="text-2xl">{cityFullInfo.socio_data.Country}</h1>
        </div>
        <div> <div  className='m-[20px] p-[15px]'>
           <div  className='w-[70vw] mx-auto'>
           <GoogleAnother data={tempData}/>
         
           </div>
           <div>
            
          
           
           </div>
        </div></div>
      </div>
    </div>
  );
};

export default User;
