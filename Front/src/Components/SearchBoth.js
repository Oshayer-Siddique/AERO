import * as React from "react";
import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";
import Autocomplete from "@mui/material/Autocomplete";
import { useContext, useState, useEffect } from "react";
import { Context } from "../App";
import axios from "axios";
import { Button } from "flowbite-react";
import { Link } from "react-router-dom";

export default function SearchBoth(props) {
  const [userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo] = useContext(Context);
//   console.log(allCountries);
  const [cityList, setCityList] = useState([
    "Sydney",
    "Newcastle",
    "Launceston",
    "Hobart",
    "Darwin",
    "Melbourne",
    "Canberra",
    "Adelaide",
    "Brisbane",
    "Perth",
    "Wollongong",
  ]);
  const [selectedCountry, setSelectedCountry] = useState({
    isSelected: false,
    country: "",
  });
  const [selectedCity, setSelectedCity] = useState({
    isSelected: false,
    city: "",
  });

  // checking focus of the input field on country
  const handleBlurCountry = (event) => {
    setSelectedCountry({
      isSelected: true,
      country: event.target.value,
    });
    // console.log(event.target.value);
  };
  const handleBlurCity = (event) => {
    setSelectedCity({
      isSelected: true,
      city: event.target.value,
    });
    // console.log(event.target.value);
  };

  // getting the states :

  useEffect(() => {
    if (selectedCountry.isSelected === true) {
      // console.log("use effect is called");
      //    var tempCity = countriesList.filter((country) => {
      //     return country.Country === selectedCountry.country;

      // })
      console.log(countriesList)
      for(let country in countriesList){
        if(countriesList[country].Country === selectedCountry.country){
          setCityList(countriesList[country].Cities);
          console.log(countriesList[country].Cities);
        }
      }
      // setCityList(tempCity[0].Cities);
      // console.log(tempCity[0].Cities);
    }
  }, [selectedCountry, countriesList]);

 
  // getting the cities :
  const handleSubmit = () => {
    var temp = {
      isAvailable: true,
      data :{
        country: selectedCountry.country,
        city: selectedCity.city,
      }
     
    };
    props.data(temp);
  };

  return (
    <div id="search-component">
      <Stack spacing={2} sx={{ width: 300 }} className="mb-10">
        <Autocomplete
          onBlur={handleBlurCountry}
          id="free-solo-demo"
          freeSolo
          options={allCountries.map((option) => option)}
          renderInput={(params) => <TextField {...params} label="Country" />}
        />
        <Autocomplete
          freeSolo
          id="free-solo-2-demo"
          onBlur={handleBlurCity}
          options={cityList.map((option) => option)}
          renderInput={(params) => <TextField {...params} label="City" />}
        />
      </Stack>

     
        <Button
          className="w-20 mx-auto mt-10"
          onClick={handleSubmit}
          variant="contained"
        >
          submit
        </Button>
     
    </div>
  );
}

// Top 100 films as rated by IMDb users. http://www.imdb.com/chart/top
