import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
// import megadata from './MegaList (1)'
import {useState, useEffect} from 'react'
import { useContext } from "react";
import { Context } from "../App";

export default function TableComponent(props) {
  const  [userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo] =
  useContext(Context);


   

  const [red,setRed] = useState(props.data)
  var a = "table-body-red"
  var b = "table-body-green"
  return (
   <div id='table-component'>
     <TableContainer component={Paper} className='w-[70vw] mx-auto'>
      <Table sx={{ minWidth: 650 }} aria-label="simple table">
        <TableHead>
          <TableRow>
            <TableCell>Rank</TableCell>
            <TableCell align="right">City</TableCell>
            <TableCell align="right">Country</TableCell>
            <TableCell align="right">AQI(chineese)</TableCell>
            <TableCell align="right">Pm25</TableCell>
            <TableCell align="right">Pm10</TableCell>
            <TableCell align="right">O3</TableCell>

          </TableRow>
        </TableHead>
        <TableBody id={red?a:b}>
          {props.json.map((row,id) => (
            <TableRow
              key={row.city}
              sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {id+1}
              </TableCell>
              <TableCell align="right">{row.city}</TableCell>
              <TableCell align="right">{row.socio_data.Country}</TableCell>
              <TableCell align="right">{row.air_data.list[0].main.aqi}</TableCell>
              <TableCell align="right">{row.air_data.list[0].components.pm2_5}</TableCell>
              <TableCell align="right">{row.air_data.list[0].components.pm10}</TableCell>
              <TableCell align="right">{row.air_data.list[0].components.o3}</TableCell>
              
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
   </div>
  );
}