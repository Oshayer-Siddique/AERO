
import {getFirestore} from "@firebase/firestore"
import { initializeApp } from "firebase/app";
const firebaseConfig = {
    apiKey: "AIzaSyAjEu9HJ4IBdimQhAR83DbRI93ZgGQHTZ8",
    authDomain: "hackathon-du.firebaseapp.com",
    projectId: "hackathon-du",
    storageBucket: "hackathon-du.appspot.com",
    messagingSenderId: "13943207907",
    appId: "1:13943207907:web:afbf14e322c90c769512d6",
    measurementId: "G-MJ93E0PS0C"
  };
  const app = initializeApp(firebaseConfig)
  const db = getFirestore(app)
  export  {app,db};