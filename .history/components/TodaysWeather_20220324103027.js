import moment from "moment-timezone";
import React from "react";
import Image from "next/image";
import LightRain from "../public/images/LightRain.png";
import NoRain from "../public/images/NoRain.png";

export default function TodaysWeather({ city, weather, timezone ,rainfall}) {
  var PathIm
  var Description
  if(rainfall<0){
    rainfall=0;
    PathIm=NoRain;
    Description='overcast clouds'
  }
  else if(rainfall<1 && rainfall>0)
  {
    PathIm=NoRain;
    Description='overcast clouds'
  }
  else if(rainfall<10 && rainfall>=1)
  {
    Description='Light Rain'
    PathIm=LightRain
  }
  else
  {
    Description='heavy intensity rain'
    PathIm=LightRain
  }
  return (
    <div className="today">
      <div className="today__inner">
        <div className="today__left-content">
          <h1>
            {city.name} ({city.country})
          </h1>

          <h2>
            <span>{weather.temp.max.toFixed(0)}&deg;C</span>
            <span>{weather.temp.min.toFixed(0)}&deg;C</span>
          </h2>

          <div className="today__sun-times">
            <div>
              <span>Sunrise</span>
              <span>
                {moment.unix(weather.sunrise).tz(timezone).format("LT")}
              </span>
            </div>

            <div>
              <span>Sunset</span>
              <span>
                {moment.unix(weather.sunset).tz(timezone).format("LT")}
              </span>
            </div>
            <div>
              <span style={{paddingLeft:"50px"}}>Rainfall</span>
              <span style={{paddingLeft:"50px"}}>
                {rainfall.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="today__right-content">
          <div className="today__icon-wrapper">
            <div>
              <Image
                src={PathIm}
                alt="Weather Icon"
                layout="fill"
              />
            </div>
          </div>

          <h3>{Description}</h3>
        </div>
      </div>
    </div>
  );
}
