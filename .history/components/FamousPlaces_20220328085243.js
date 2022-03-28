import React from "react";
import Image from "next/image";
import Link from "next/link";

// import images
import HaNoi from "../public/images/HaNoi.jpeg";
import HaTay from "../public/images/HaTay.jpeg";
import HaTinh from "../public/images/HaTinh.jpeg";
import HaGiang from "../public/images/HaGiang.jpeg";

const places = [
  {
    name: "Thủ Đô Hà Nội",
    image: LondonImage,
    url: "/location/Thủ-Đô-Hà-Nội-1581129",
  },
  {
    name: "Tỉnh Hà Giang",
    image: ParisImage,
    url: "/location/Tỉnh-Hà-Giang-1581030",
  },
  {
    name: "Tỉnh Hà Tây",
    image: TokyoImage,
    url: "/location/Tỉnh-Hà-Tây-1581019",
  },
  {
    name: "Tỉnh Hà Tĩnh",
    image: NewYorkImage,
    url: "/location/Tỉnh-Hà-Tĩnh-1580700",
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
