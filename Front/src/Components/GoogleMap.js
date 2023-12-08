import React, { Component } from 'react';
import { Map, GoogleApiWrapper, Marker } from 'google-maps-react';
import { useContext, useState, useEffect } from "react";
import { Context } from "../App";
class GoogleMap extends Component {
  render() {
    const { data } = this.props;
    // console.log(data.address.current.pollution.aqius);
    const initialPosition = {
      lat: data.address.location.coordinates[1], // Latitude of your initial map position
      lng: data.address.location.coordinates[0], // Longitude of your initial map position
    };
    const customMapStyles = [
        {
          featureType: 'administrative',
          elementType: 'all',
          stylers: [{ saturation: -100 }],
        },
        {
          featureType: 'administrative.province',
          elementType: 'all',
          stylers: [{ visibility: 'off' }],
        },
        {
          featureType: 'landscape',
          elementType: 'all',
          stylers: [
            { saturation: -100 },
            { lightness: 65 },
            { visibility: 'on' },
          ],
        },
        {
          featureType: 'poi',
          elementType: 'all',
          stylers: [
            { saturation: -100 },
            { lightness: '50' },
            { visibility: 'simplified' },
          ],
        },
        {
          featureType: 'road',
          elementType: 'all',
          stylers: [{ saturation: -100 }],
        },
        {
          featureType: 'road.highway',
          elementType: 'all',
          stylers: [{ visibility: 'simplified' }],
        },
        {
          featureType: 'road.arterial',
          elementType: 'all',
          stylers: [{ lightness: '30' }],
        },
        {
          featureType: 'road.local',
          elementType: 'all',
          stylers: [{ lightness: '40' }],
        },
        {
          featureType: 'transit',
          elementType: 'all',
          stylers: [
            { saturation: -100 },
            { visibility: 'simplified' },
          ],
        },
        {
          featureType: 'water',
          elementType: 'geometry',
          stylers: [
            { hue: '#ffff00' },
            { lightness: -25 },
            { saturation: -97 },
          ],
        },
        {
          featureType: 'water',
          elementType: 'labels',
          stylers: [
            { lightness: -25 },
            { saturation: -100 },
          ],
        },
      ];
      
    return (
      <Map
        google={this.props.google}
      
        initialCenter={initialPosition}
        zoom={12}
        styles={customMapStyles}
      >
        <Marker
        className="bg-green-500"
          position={initialPosition}
          title={ data.address.city} // Change to your desired marker title
          name={data.address.current.pollution.aqius} // Change to your desired marker name
        />
      </Map>
    );
  }
}

export default GoogleApiWrapper({
  apiKey: 'AIzaSyCf3mY9G1oCiVocrVgVGA0eeDRW2dUG_Wg', // Replace with your actual API key
})(GoogleMap);
