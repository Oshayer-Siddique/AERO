const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const cookieParser = require("cookie-parser");
const multer = require('multer');
const path = require('path');
const dotenv = require('dotenv');
const cors = require('cors');
const countries = require("i18n-iso-countries");
var axios = require('axios');


const app = express();

app.use(cors({
    origin: "http://localhost:3000",
}));
app.use(express.json())
app.use(express.urlencoded({ extended: true }));
dotenv.config();





const router1 = require('./router/router1');

app.use('/',router1);




app.listen(process.env.port,()=> {
    console.log(`server listening on port ${process.env.port}`)
})