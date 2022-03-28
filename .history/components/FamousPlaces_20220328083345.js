import React from "react";
import Image from "next/image";
import Link from "next/link";

// import images
import LondonImage from "../public/images/london.jpg";
import ParisImage from "../public/images/paris.jpg";
import TokyoImage from "../public/images/tokyo.jpg";
import NewYorkImage from "../public/images/new-york.jpg";

const places = [
  {
    name: "Thủ Đô Hà Nội",
    image: LondonImage,
    url: "/location/Thủ-Đô-Hà-Nội-2643743",
  },
  {
    name: "Tỉnh Hà Giang",
    image: ParisImage,
    url: "/location/Tỉnh-Hà-Giang-2968815",
  },
  {
    name: "TỈnh Hà Tây",
    image: TokyoImage,
    url: "/location/Tỉnh-Hà-Tây-1850147",
  },
  {
    name: "Tỉnh Hà Tĩnh",
    image: NewYorkImage,
    url: "/location/Tỉnh-Hà-Tĩnh-5128581",
  },
];

export default function FamousPlaces() {
  return (
    <div className="places">
      <div className="places__row">
        {places.length > 0 &&
          places.map((place, index) => (
            <div className="places__box" key={index}>
              <Link href={place.url}>
                <a>
                  <div className="places__image-wrapper">
                    <Image
                      src={place.image}
                      alt={`${place.name} Image`}
                      layout="fill"
                      objectFit="cover"
                    />
                  </div>

                  <span>{place.name}</span>
                </a>
              </Link>
            </div>
          ))}
      </div>
    </div>
  );
}
