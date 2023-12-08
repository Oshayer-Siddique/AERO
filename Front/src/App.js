import logo from './logo.svg';
import './App.css';
import { Link, Route, Routes } from 'react-router-dom';
import Home from './Pages/Home';
import Navbar from './Components/Navbar';
import Login from './Pages/Login';
import { useEffect, useState } from 'react';
import { createContext } from 'react';
import User from './Pages/User';
import axios from 'axios'
import FormData from 'form-data';

// import megadata from '../src/Components/MegaList (1)'
export const Context = createContext(); 
function App() {

  // city full info : 
  const [cityFullInfo,setCityFullInfo] = useState({
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


  const [userDetails,setUser]=useState({
    isSignedIn: false,
    userName:"",
    UserEmail :"",
    UserImage :""
  });
  
// getting the list of all countries 
const [countriesList,setCountriesList]=useState([
  {
  "Country": "Afghanistan",
  "Cities": [
  "Kabul"
  ]
  },
  {
  "Country": "Algeria",
  "Cities": [
  "Algiers"
  ]}]);

useEffect(()=>{
  
var config = {
  method: 'get',
maxBodyLength: Infinity,
  url: 'http://34.42.153.40:5000/countrylist',
  headers: { 
    'Content-Type': 'application/json'
  }

};

axios(config)
.then(function (response) {
  setCountriesList(response.data);
  // console.log(countriesList);
})
.catch(function (error) {
  console.log(error);
});

},[]);

const [megadata,setMegaData] = useState( [
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
useEffect(()=>{
  
  var config = {
    method: 'get',
  maxBodyLength: Infinity,
    url: 'http://34.42.153.40:5000/getalldata',
    headers: { 
      'Content-Type': 'application/json'
    }
  
  };
  
  axios(config)
  .then(function (response) {
    setMegaData(response.data);
  })
  .catch(function (error) {
    console.log(error);
  });
  
  },[]);
// getting countries : 
const [allCountries,setAllCountries] = useState([])
useEffect(()=>{
  var countries = countriesList.map((dt)=>{
    return dt.Country
  })
  setAllCountries(countries)
  // console.log(countries);
}
,[countriesList])


const [areaDetatils,setAreaDetails] = useState({
   isAvailable: false,
  data :{
    country: "",
    city: "",
  }})


// getting the address form the ip address
const [userAddress,setUserAddress] = useState({
  isAvailable:false,
  address:{
    "isAvailable": true,
    "address": {
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
    }
},
})

// getting address : 

useEffect(()=>{
  
  // console.log("use effect is called");
  
  var config = {
    method: 'get',
  maxBodyLength: Infinity,
    url: 'http://api.airvisual.com/v2/nearest_city?key=84f0b84b-a9e4-44d9-9726-dc8755b33c9f',
    headers: { 
      'Content-Type': 'application/json'
    }
  
  };
  
  axios(config)
  .then(function (response) {
    setUserAddress({
      isAvailable:true,
      address:response.data.data});
    // console.log(response.data.data);
  })
  .catch(function (error) {
    console.log(error);
  });
  
  },[]);


  // getting all sorted cities : 
  
//   const [sortedCities,setSortedCities] = useState([
//     {
//     "City": "Gasteiz / Vitoria",
//     "Coordinates": [
//     42.84998,
//     -2.67268
//     ],
//     "aqi": 2,
//     "Country": "Spain"
//     },])
// useEffect(()=>{
  
//   console.log("use effect is called");
  
//   var config = {
//     method: 'get',
//   maxBodyLength: Infinity,
//     url: 'http://34.42.153.40:5000/sortcity',
//     headers: { 
//       'Content-Type': 'application/json'
//     }
  
//   };
  
//   axios(config)
//   .then(function (response) {
//     setCountriesList(response.data);
//     console.log(countriesList);
//   })
//   .catch(function (error) {
//     console.log(error);
//   });
  
//   },[]);

  return (
    <Context.Provider value={[userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo,megadata]}>
    <div className="App">
      <Navbar/>
      <Routes>
      <Route path='/' element={<Home/>}/>
      <Route path='/Login' element={<Login/>}/>
      <Route path='/User' element={<User/>}/>
    </Routes>
    </div>
    </Context.Provider>
  );
}

export default App;
