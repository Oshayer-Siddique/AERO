import * as React from 'react';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import "firebase/auth"
import {app,db} from './firebase.config';
import {signOut, getAuth, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import {useState} from 'react';
import { useContext,useEffect } from 'react';
import { Context } from '../App';
// firestore import
import { collection, addDoc,getDocs } from "firebase/firestore";
import GoogleMap from '../Components/GoogleMap';
import GoogleAnother from '../Components/GoogleAnother';
import { Chart } from '../Components/Chart';
import GoogleMany from '../Components/GoogleMany';
export default function Login() {
const [userDetails, setUser, countriesList,areaDetatils,setAreaDetails,userAddress] = useContext(Context);
const defaultTheme = createTheme();

const provider = new GoogleAuthProvider()
  const handleSubmit = (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    console.log({
      email: data.get('email'),
      password: data.get('password'),
    });
  };
  var signIn =()=>{
    const auth = getAuth()
    signInWithPopup(auth,provider)
    .then(result=>{
      const credential = GoogleAuthProvider.credentialFromResult(result);
    const token = credential.accessToken;
     const user = result.user;
  // a object is made with the user data 
     var userDetails={
      isSignedIn: true,
    userName:user.displayName,
    UserEmail :user.email,
    UserImage :user.photoURL
     }
  // user object is being stored into useState
     setUser(userDetails)
    })

  // handling error for auth
    .catch((error) => {
     
      const errorCode = error.code;
      const errorMessage = error.message;      
      const email = error.customData.email;      
      const credential = GoogleAuthProvider.credentialFromError(error);
      console.log(`error code = ${errorCode}`)
      console.log(`error message = ${errorMessage}`)
      console.log(`error email = ${email}`)
      console.log(`credential  = ${credential}`)

      
    });
  }

// this function take care of signOut
var signout =()=>{
  
      const auth = getAuth();
      signOut(auth).then(() => {
        console.log("sign out")
      // here it reinitialize the object to put into setUser
        var userDetails={
        isSignedIn: false,
        userName:"",
        UserEmail :"",
        UserImage :""
         }
      setUser(userDetails)
      })
      .catch((error) => {
        console.log(error)
      });
}


// sending user data to database : 

const userCollectionRef = collection(db, "users");
console.log(userAddress)
useEffect(()=>{
 const getUsers = async ()=>{

  if (userDetails.isSignedIn) {
    await addDoc(userCollectionRef,userDetails)
    
  }

 }
   
 
 getUsers()

  
},[userDetails])


return (
    <div className="App">
      {/* {
        userDetails.isSignedIn?
        <button className='border-4 p-4 rounded-md bg-green-500' onClick={signout} >sign out </button>:
        <button className='border-4 p-4 rounded-md bg-green-500' onClick={signIn} >sign in </button>
      } */}
     
    {/* {
      userDetails.isSignedIn? 
      <div>
        <h1>{userDetails.userName}</h1>
        <h2>{userDetails.UserEmail}</h2>
        <img src={userDetails.UserImage} alt="" />
        </div>
        :
        <>
          <h1>Not Logged in </h1>
          {/* <GoogleMap data={userAddress}/> */}
         
          {/* <GoogleMany/> */}
          
      
 <GoogleAnother/>
 <Chart/>
    </div>
  );
  }