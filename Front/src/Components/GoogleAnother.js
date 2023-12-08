import React, { useEffect, useState } from 'react';

function GoogleAnother({data}) {
  // console.log(data);
  const [showData,setShowData] = useState({})
  useEffect(() => {
  
    try { 
    if (data.isAvailable === true) {
        var temp = {
          city : data.address.city,
          country : data.address.country,
          coordinateX : data.address.location.coordinates[1],
          coordinateY : data.address.location.coordinates[0],
          aqi : data.address.current.pollution.aqius

                }

          setShowData(temp)
          // console.log(showData)
      
    }
  }catch (error) {
    console.log(error);
  } }, [data]);



  useEffect(() => {
  async  function initMap() {
      const position = { lat: showData.coordinateX, lng: showData.coordinateY};

        // Import the Marker Library
        const { AdvancedMarkerElement } = await window.google.maps.importLibrary('marker');
        const {PinElement} = await window.google.maps.importLibrary("marker")
      const map = new window.google.maps.Map(document.getElementById('map'), {
        zoom: 4,
        center: position,
        mapId: 'dffd490edc447297',
        disableDefaultUI: true,
      });
      const pinScaled = new PinElement({
        scale: 3.5,
        background: "#FBBC04",
      });
      const priceTag = document.createElement("div");

      priceTag.className = "price-tag";
      priceTag.innerHTML = `City : ${showData.city}<br/>Country : ${showData.country}<br/>AQI : ${showData.aqi}`
  
      const marker = new AdvancedMarkerElement({
        map: map,
        position: position,
        title: 'Uluru',
        content: priceTag,
      });
    }

    // Make sure the Google Maps API is loaded
    if (window.google) {
      initMap();
    }
  }, [showData]);

  return (
    <div id="map" style={{ width: '60%', height: '400px' }} />
  );
}

export default GoogleAnother;
