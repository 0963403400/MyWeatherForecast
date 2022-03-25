import React from "react";
import moment from "moment-timezone";
import Image from "next/image";
import LightRain from "../public/images/LightRain.png";
import NoRain from "../public/images/NoRain.png";

export default function WeeklyWeather({ weeklyWeather, timezone ,rainfall}) {
  // console.log(rainfall);
  var PathIm
  var Description
  
  return (
    <div className="weekly">
      <h3 className="weekly__title">
        Weekly <span>Weather</span>
      </h3>

      {weeklyWeather.length > 0 &&
        weeklyWeather.map((weather, index) => {
          if (index == 0 || index==7) {
            return;
          }
          var RainFallValue;
          switch(index){
            case 1:
              RainFallValue=rainfall.DAY2;
              break;
            case 2:
              RainFallValue=rainfall.DAY3;
              break;
            case 3:
              RainFallValue=rainfall.DAY4;
              break;
            case 4:
              RainFallValue=rainfall.DAY5;
              break;
            case 5:
              RainFallValue=rainfall.DAY6;
              break;
            case 6:
              RainFallValue=rainfall.DAY7;
              break;
          }
          if(RainFallValue<0){
            RainFallValue=0;
            PathIm=NoRain;
            Description='overcast clouds'
          }
          else if(RainFallValue<1 && RainFallValue>0)
          {
            PathIm=NoRain;
            Description='overcast clouds'
          }
          else if(RainFallValue<10 && RainFallValue>=1)
          {
            Description='Light Rain'
            PathIm=LightRain
          }
          else
          {
            Description='heavy intensity rain'
            PathIm=LightRain
          }
          // console.log(weather.weather[0].icon)
          return (
            <div className="weekly__card" key={weather.dt}>
              <div className="weekly__inner">
                <div className="weekly__left-content">
                  <div>
                    <h3>
                      {moment.unix(weather.dt).tz(timezone).format("dddd")}
                    </h3>

                    <h4>
                      <span>{weather.temp.max.toFixed(0)}&deg;C</span>
                      <span>{weather.temp.min.toFixed(0)}&deg;C</span>
                    </h4>
                  </div>

                  <div className="weekly__sun-times">
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
                      <span style="paddingLeft:30px">RainFall</span>
                      <span>
                        {RainFallValue}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="weekly__right-content">
                  <div className="weekly__icon-wrapper">
                    <div>
                      <Image
                        // src={`https://openweathermap.org/img/wn/${weather.weather[0].icon}@2x.png`}
                        src={PathIm}
                        alt="Weather Icon"
                        layout="fill"
                      />
                    </div>
                  </div>

                  <h5>        {Description}</h5>
                </div>
              </div>
            </div>
          );
        })}
    </div>
  );
}
