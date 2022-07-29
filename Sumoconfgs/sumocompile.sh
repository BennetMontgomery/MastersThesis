#!/usr/local/bin/bash

# load networks
nets=()
for file in ./*.net.xml
do
  nets+=("$file")
done

# load routes
declare -A routes=()
for file in ./*.rou.xml
do
  routes["$file"]="$file"
done

for netfile in ${nets[*]}
do
  tmp1="${netfile#*/}"
  tmp2="${tmp1%.*}"
  name="${tmp2%.*}"

  # check if network file has an equivalent route file
  if [[ -n ${routes["./$name.rou.xml"]} ]]
  then
    echo -e "<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>" > "$name.sumocfg"
    echo -e "<configuration xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" >> "$name.sumocfg"
    echo -e "\txsi:noNameSpaceSchemaLocation=\"http://sumo.dlr.de/xsd/sumoConfiguration.xsd\">" >> "$name.sumocfg"
    echo -e "\t<input>" >> "$name.sumocfg"
    echo -e "\t\t<net-file value=\"$name.net.xml\"/>" >> "$name.sumocfg"
    echo -e "\t\t<route-files value=\"$name.rou.xml\"/>" >> "$name.sumocfg"
    echo -e "\t</input>" >> "$name.sumocfg"
    echo -e "\t<time>" >> "$name.sumocfg"
    echo -e "\t\t<begin value=\"0\"/>" >> "$name.sumocfg"
    echo -e "\t\t<end value=\"1000\"/>" >> "$name.sumocfg"
    echo -e "\t</time>" >> "$name.sumocfg"
    echo -e "\t<time-to-teleport value=\"-1\"/>" >> "$name.sumocfg"
    echo -e "</configuration>" >> "$name.sumocfg"
  fi
done