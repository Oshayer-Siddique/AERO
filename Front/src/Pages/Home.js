import React, { useState } from 'react';
import UploadComponent from '../Components/UploadComponent';
import TableComponent from '../Components/TableComponent';
import SearchComponent from '../Components/SearchComponent';
import GoogleAnother from '../Components/GoogleAnother';
import { Chart } from '../Components/Chart';
import GoogleMany from '../Components/GoogleMany';
import { useContext, useEffect } from "react";
import { Context } from "../App";
import SearchBoth from '../Components/SearchBoth';

const Home = () => {
    const[userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo,megadata] =
    useContext(Context);

    const [firstSearch,setFirstSearch] = useState({
        isAvailable: false,
       data :{
         country: "",
         city: "",
       }})
    const [lastSearch,setLastSearch] = useState({
        isAvailable: false,
       data :{
         country: "",
         city: "",
       }})
       const [first_ten,setFirst_ten] = useState([])
       const [last_ten,setLast_ten] = useState([])
       
       const [rows,setRow] =useState( [
         {
             "status": "success",
             "message": "data is Ok",
             "city": "Algiers",
             "air_data": {
                 "coord": {
                     "lon": 3.09,
                     "lat": 36.73
                 },
                 "list": [
                     {
                         "main": {
                             "aqi": 2
                         },
                         "components": {
                             "co": 220.3,
                             "no": 0,
                             "no2": 5.31,
                             "o3": 86.55,
                             "so2": 0.74,
                             "pm2_5": 0.7,
                             "pm10": 1.96,
                             "nh3": 0.09
                         },
                         "dt": 1699049290
                     }
                 ]
             },
             "socio_data": {
                 "population": 44903225,
                 "populationGrowth": 1.62833925417918,
                 "GDP": 191912889420.911,
                 "GDPPerCapita": 4273.922183115155,
                 "GDPGrowth": 3.09999999986282,
                 "Country": "Algeria"
             }
         }])
       
         //console.log(megadata[2].air_data.list[0].main.aqi)
         useEffect(()=>{
           let mdata = [...megadata]
           for(let i=0; i<mdata.length - 1; i++){
             for(let j=i+1; j<mdata.length; j++){
               if(mdata[i].air_data.list[0].components.pm10 > mdata[j].air_data.list[0].components.pm10){
                 let temp = mdata[i]
                 mdata[i] = mdata[j]
                 mdata[j] = temp
               }
             }
           }
           for(let i=0; i<mdata.length - 1; i++){
             for(let j=i+1; j<mdata.length; j++){
               if(mdata[i].air_data.list[0].components.pm2_5 > mdata[j].air_data.list[0].components.pm2_5){
                 let temp = mdata[i]
                 mdata[i] = mdata[j]
                 mdata[j] = temp
               }
             }
           }
           for(let i=0; i<mdata.length - 1; i++){
             for(let j=i+1; j<mdata.length; j++){
               if(mdata[i].air_data.list[0].main.aqi > mdata[j].air_data.list[0].main.aqi){
                 let temp = mdata[i]
                 mdata[i] = mdata[j]
                 mdata[j] = temp
               }
             }
           }
         
           
         setFirst_ten(mdata.slice(0,10)) 
         setLast_ten( mdata.slice(mdata.length - 10, mdata.length).reverse())
         },[]) 
    return (
        <div id='home' >
            <div>
                <h1 id='home-main-line'>City Air Quality at Your Fingertips</h1>
                <h2 id='home-second-line'>Real-time data on city air quality at your fingertips</h2>
            </div>
        <div  className='m-[20px] p-[15px]'>
           <div id='map-search' className='flex flex-col md:flex-row '>
           <GoogleAnother data={userAddress}/>
         
           
           <SearchComponent className='self-center'/>
        
          
           </div>
           <div>
          
           
           </div>
        </div>
         
            <div id='record' className='m-10 rounded-md p-20'>
                <h1 className='text-2xl mb-10'>
                    Most Polluted Cities
                </h1>
                <TableComponent data={true} json={last_ten}/>
            </div>
            <div className='m-10 rounded-md p-20'>
                <h1 className='text-2xl mb-10 '>
                   Least Polluted Cities
                </h1>
                <TableComponent data={false} json={first_ten}/>
            </div>
            
         
          <div id='compare'>
          <h1 className='text-2xl font-semibold mb-10'>Compare Two Cities</h1>
            <div id='search-two-box' className='flex justify-between w-[70vw] mx-auto'>
           
            <SearchBoth data={setFirstSearch}  className='self-center '/>
            <SearchBoth data={setLastSearch}  className='self-center '/>
            </div>
            <div className='mb-10 '>
                <Chart data={[firstSearch,lastSearch]}/>
            </div>
            <h1 className='text-2xl font-semibold'>Find All Cities Here</h1>
            <div >
                <GoogleMany/>
            </div>
          </div>


        </div>
    );
};

export default Home;