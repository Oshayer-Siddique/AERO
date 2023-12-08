import React, { useEffect } from 'react';
import { useContext, } from "react";
import { Context } from "../App";

function GoogleMany() {
  const[userDetails,setUser,countriesList,areaDetatils,setAreaDetails,userAddress,allCountries,cityFullInfo,setCityFullInfo,megadata] =
useContext(Context);
const hundred = megadata
  useEffect(() => {
  async  function initMap() {
     // Request needed libraries.
  const { Map } = await window.google.maps.importLibrary("maps");
  const { AdvancedMarkerElement } = await window.google.maps.importLibrary("marker");
  const { LatLng } = await window.google.maps.importLibrary("core");
  const center = new LatLng(37.43238031167444, -122.16795397128632);
  const map = new Map(document.getElementById("mapmany"), {
    zoom: 5,
    center,
        mapId: 'dffd490edc447297',
        disableDefaultUI: true,
      });

      for (const property of hundred) {
        const AdvancedMarkerElement = new window.google.maps.marker.AdvancedMarkerElement({
          map,
          content: buildContent(property),
          position:  {
            lat: property.air_data.coord.lat,
            lng: property.air_data.coord.lon,
          },
          title: property.description,
        });
    
        AdvancedMarkerElement.addListener("click", () => {
          toggleHighlight(AdvancedMarkerElement, property);
        });
      }
    }
  
    function toggleHighlight(markerView, property) {
      if (markerView.content.classList.contains("highlight")) {
        markerView.content.classList.remove("highlight");
        markerView.zIndex = null;
      } else {
        markerView.content.classList.add("highlight");
        markerView.zIndex = 1;
      }
    }
    
    function buildContent(property) {
      const content = document.createElement("div");
    
      content.classList.add("property");
      content.innerHTML = `
        <div class="icon">
         
            <span class="fa-sr-only">${property.city}</span>
        </div>
        <div class="details">
            <div class="price">${property.city}</div>
            <div class="address">${property.socio_data.Country}</div>
            <div class="features">
            <div>
                <i aria-hidden="true" class="fa fa-bed fa-lg bed" title="bedroom"></i>
                <span class="fa-sr-only">Population</span>
                <span>${property.socio_data.population}</span>
            </div>
            <div>
                <i aria-hidden="true" class="fa fa-bath fa-lg bath" title="bathroom"></i>
                <span class="fa-sr-only">GDP</span>
                <span>${property.socio_data.GDP}</span>
            </div>
            <div>
                <i aria-hidden="true" class="fa fa-ruler fa-lg size" title="size"></i>
                <span class="fa-sr-only">AQI</span>
                <span>${property.air_data.list[0].main.aqi} </span>
            </div>
            </div>
        </div>
        `;
      return content;
    }
    // Make sure the Google Maps API is loaded
    if (window.google) {
      initMap();
    }
  }, []);

  return (
    <div id="mapmany" style={{ width: '100%', height: '400px' }} />
  );
}
const properties = [
  {
    city: "215 Emily St, MountainView, CA",
    description: "Single family house with modern design",
    price: "$ 3,889,000",
    type: "home",
    bed: 5,
    bath: 4.5,
    size: 300,
    position: {
      lat: 37.50024109655184,
      lng: -122.28528451834352,
    },
  },
  {
    city: "108 Squirrel Ln &#128063;, Menlo Park, CA",
    description: "Townhouse with friendly neighbors",
    price: "$ 3,050,000",
    type: "building",
    bed: 4,
    bath: 3,
    size: 200,
    position: {
      lat: 37.44440882321596,
      lng: -122.2160620727,
    },
  },
  {
    city: "100 Chris St, Portola Valley, CA",
    description: "Spacious warehouse great for small business",
    price: "$ 3,125,000",
    type: "warehouse",
    bed: 4,
    bath: 4,
    size: 800,
    position: {
      lat: 37.39561833718522,
      lng: -122.21855116258479,
    },
  },
  {
    city: "98 Aleh Ave, Palo Alto, CA",
    description: "A lovely store on busy road",
    price: "$ 4,225,000",
    type: "store-alt",
    bed: 2,
    bath: 1,
    size: 210,
    position: {
      lat: 37.423928529779644,
      lng: -122.1087629822001,
    },
  },
  {
    city: "2117 Su St, MountainView, CA",
    description: "Single family house near golf club",
    price: "$ 1,700,000",
    type: "home",
    bed: 4,
    bath: 3,
    size: 200,
    position: {
      lat: 37.40578635332598,
      lng: -122.15043378466069,
    },
  },
  {
    city: "197 Alicia Dr, Santa Clara, CA",
    description: "Multifloor large warehouse",
    price: "$ 5,000,000",
    type: "warehouse",
    bed: 5,
    bath: 4,
    size: 700,
    position: {
      lat: 37.36399747905774,
      lng: -122.10465384268522,
    },
  },
  {
    city: "700 Jose Ave, Sunnyvale, CA",
    description: "3 storey townhouse with 2 car garage",
    price: "$ 3,850,000",
    type: "building",
    bed: 4,
    bath: 4,
    size: 600,
    position: {
      lat: 37.38343706184458,
      lng: -122.02340436985183,
    },
  },
  {
    city: "868 Will Ct, Cupertino, CA",
    description: "Single family house in great school zone",
    price: "$ 2,500,000",
    type: "home",
    bed: 3,
    bath: 2,
    size: 100,
    position: {
      lat: 37.34576403052,
      lng: -122.04455090047453,
    },
  },
  {
    city: "655 Haylee St, Santa Clara, CA",
    description: "2 storey store with large storage room",
    price: "$ 2,500,000",
    type: "store-alt",
    bed: 3,
    bath: 2,
    size: 450,
    position: {
      lat: 37.362863347890716,
      lng: -121.97802139023555,
    },
  },
  {
    city: "2019 Natasha Dr, San Jose, CA",
    description: "Single family house",
    price: "$ 2,325,000",
    type: "home",
    bed: 4,
    bath: 3.5,
    size: 500,
    position: {
      lat: 37.41391636421949,
      lng: -121.94592071575907,
    },
  },
];

export default GoogleMany;
