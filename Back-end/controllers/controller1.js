const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const cookieParser = require("cookie-parser");
const multer = require("multer");
const path = require("path");
var axios = require('axios');
const fs = require('fs').promises;
const fs1 = require('fs')
const countries = require("i18n-iso-countries");




let lat, lon;


async function getcoordinate(req, res) {
    const { country, city } = req.body;
    //console.log(req.body.city);


    // res.send(req.body.country);
    const apiUrl2 = './controllers/geo_citi.json';

    let datta = await fs.readFile(apiUrl2, 'utf8')
    //console.log(datta)

    try {
        const jsonData = JSON.parse(datta);
        //console.log(jsonData);

        for (let item of jsonData) {
            if (item["City"] == req.body.city) {
                lat = (item["Coordinates"])[0];
                lon = (item["Coordinates"])[1];
                lat = lat.toFixed(2);
                lon = lon.toFixed(2);
                const u = [];
                u.push(lat, lon);
                //console.log(u);
            }
        }
        //res.send("oK");


    }
    catch (err) {
        console.log("Error");
    }

    const apiUrl3 = `http://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lon}&appid=${process.env.api_key1}`;

    //console.log(apiUrl3);
    const response1 = await axios.get(apiUrl3);
    //console.log(response1.data);

    //Get SocioEco Stuff

    // J.M Areeb Uzair
    // const countries = require("i18n-iso-countries");
    // const axios = require("axios");


    //Get Country Code
    // const code = countries.getAlpha2Code(req.body.country, "en");
    // sociodata = {};
    // //console.log(code)
    // let A, B;
    // //Find Population
    // A = Date.now();
    // let socioresponse = await axios.get(`https://api.worldbank.org/v2/country/${code}/indicator/SP.POP.TOTL?format=json`);
    // B = Date.now()
    // console.log(B-A);
    // sociodata.population = socioresponse.data[1][0].value;
    // //Find Population Growth
    // A = Date.now();
    // socioresponse = await axios.get(`https://api.worldbank.org/v2/countries/${code}/indicators/SP.POP.GROW?format=json`);
    // B = Date.now()
    // console.log(B-A);
    // sociodata.populationGrowth = socioresponse.data[1][0].value;
    // //Find GDP and GDP per Capita
    // A = Date.now();
    // socioresponse = await axios.get(`https://api.worldbank.org/v2/country/${code}/indicator/NY.GDP.MKTP.CD?format=json`);
    // B = Date.now()
    // console.log(B-A);
    // sociodata.GDP = socioresponse.data[1][0].value;
    // sociodata.GDPPerCapita = socioresponse.data[1][0].value / sociodata.population;
    // //Find GDP Growth
    // A = Date.now();
    // socioresponse = await axios.get(`https://api.worldbank.org/v2/country/${code}/indicator/NY.GDP.MKTP.KD.ZG?format=json`);
    // B = Date.now()
    // console.log(B-A);
    // sociodata.GDPGrowth = socioresponse.data[1][0].value;
    // // console.log(sociodata);


    const filepath_socio = '.\\controllers\\socia_data.json';

    datta = await fs.readFile(filepath_socio, 'utf8')
    let socio_data;
    //console.log(datta)

    try {
        const jsonData = JSON.parse(datta);
        //console.log(jsonData);

        for (let item of jsonData) {
            if (item["Country"] == req.body.country) {
                socio_data = item;
                break;
                //console.log(u);
            }
        }
        //res.send("oK");


    }
    catch (err) {
        console.log("Error");
    }
    


    console.log(socio_data)


    res.json({ status: 'success', message: 'data is Ok', city: req.body.city, air_data: response1.data, socio_data: socio_data})





    //res.send("Hellooooooooo")

}

function sortByPropertyAsc(property){  
    return function(a,b){  
       if(a[property] > b[property])  
          return 1;  
       else if(a[property] < b[property])  
          return -1;  
   
       return 0;  
    }  
 }

 function sortByPropertyDsc(property){  
    return function(a,b){  
       if(a[property] < b[property])  
          return 1;  
       else if(a[property] > b[property])  
          return -1;  
   
       return 0;  
    }  
 }

async function sortcity(req,res){
    
    const filepath = './controllers/geo_citi.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            console.log("1");
            const jsonData = JSON.parse(data);
            console.log("2");
        
            // Sort the JSON data based on the "City" property
            jsonData.sort(sortByPropertyAsc("aqi"))
            console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}


async function sortcityDsc(req,res){
    
    const filepath = './controllers/geo_citi.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            //console.log("1");
            const jsonData = JSON.parse(data);
            //console.log("2");
        
            // Sort the JSON data based on the "City" property
            jsonData.sort(sortByPropertyDsc("aqi"))
            //console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}


async function sortpopulation_socialdata(req,res){
    
    const filepath = './controllers/socia_data.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            //console.log("1");
            const jsonData = JSON.parse(data);
            //console.log("2");
        
            // Sort the JSON data based on the "City" property
            jsonData.sort(sortByPropertyDsc("population"))
            //console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}


async function sortGdp_socialdata(req,res){
    
    const filepath = './controllers/socia_data.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            //console.log("1");
            const jsonData = JSON.parse(data);
            //console.log("2");
        
            // Sort the JSON data based on the "City" property
            jsonData.sort(sortByPropertyDsc("GDP"))
            //console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}


async function countrylist(req,res){
    
    const filepath = './controllers/countries.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            //console.log("1");
            const jsonData = JSON.parse(data);
            //console.log("2");
        
            // Sort the JSON data based on the "City" property
            jsonData.sort(sortByPropertyAsc("Country"))
            //console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}



async function getfulldata(req,res){
    
    const filepath = './controllers/megafile.json';
    //console.log("Begin");
    fs1.readFile(filepath,'utf8',(err,data) =>{
        if(err){
            console.error(err);
            return;
        }
        try {
            // Parse the JSON data
            console.log("1");
            const jsonData = JSON.parse(data);
            console.log("2");
        
            // Sort the JSON data based on the "City" property
            // jsonData.sort(sortByPropertyAsc("aqi"))
            // console.log(3);
        
            // Convert the sorted data back to a JSON string
            const sortedJson = JSON.stringify(jsonData, null, 2);
            res.send(sortedJson);
        }
        catch(err){

        }

          
        
    })

    


}






module.exports = {
    countrylist,
    getcoordinate,
    sortcity,
    sortcityDsc,
    sortpopulation_socialdata,
    sortGdp_socialdata,
    getfulldata,
}
