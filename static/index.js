/* ----------------------------------------------------------------*/
/* ----------------------Menu bar ---------------------------------*/
/* ----------------------------------------------------------------*/
/* ----------------------------------------------------------------*/
function menuShow(){
    var y=document.getElementById("navBarId");
    var x=document.getElementById("menuIconId");
    var z=document.getElementById("searchBarBoxContainDivId");
    if (y.style.marginLeft == "-100%") {
          y.style.marginLeft = "5%";
          x.style.color="#2ecc71";
          z.style.marginLeft="-100%"
         
        } else {
          
            y.style.marginLeft = "-100%";
            x.style.color="white";
           
            z.style.marginLeft="0%"
          
          }
          
    }
/* ----------------------------------------------------------------*/  
var inWidth1=window.innerWidth;
var element1 = document.getElementById("navBarId");
if(inWidth1<630){element1.style.marginLeft = "-100%";}
else{element1.style.marginLeft = "5%";  } 
  /* -------------------------------------------------------------- --*/ 
    function getWidth(){
      var inWidth=window.innerWidth;
      if(inWidth>572){
        var element = document.getElementById("navBarId");
        var element2=document.getElementById("searchBarBoxContainDivId");

      var attr = element.getAttributeNode("style");
      element.removeAttributeNode(attr);  

      var attr = element2.getAttributeNode("style");
      element.removeAttributeNode(attr);  

      }else{
        var element = document.getElementById("navBarId");
        var element2=document.getElementById("searchBarBoxContainDivId");
        element.style.marginLeft = "-100%";
        element2.style.marginLeft = "0%";
      }
    }



/* ----------------------------------------------------------------*/
/* --------------------set lang attribute to  the head tag-----------*/
/* ----------------------------------------------------------------*/
/* ------------------;----------------------------------------------*/ 
tarhtml = document.querySelector("html");

tarhtml.setAttribute("lang", "en");
/* ----------------------------------------------------------------*/
/* --------------------set favicon-----------*/
/* ----------------------------------------------------------------*/
/* ------------------;----------------------------------------------*/   
tarhead = document.querySelector("head");
var createlinktag = document.createElement("link"); 
createlinktag.setAttribute("rel", "icon");
createlinktag.setAttribute("href", "../../images/Coders Aim 3.JPG");
createlinktag.setAttribute("type", "image/x-icon");
tarhead.appendChild(createlinktag);


/* ----------------------------------------------------------------*/
/* -------------------------------------------------------*/
/* --------------------Menubar logo--------------------------*/
mtitle = document.querySelector(".titleBasicCss");
var titletxt= document.createTextNode("Codersaim");
mtitle.appendChild(titletxt);

/* ----------------------------------------------------------------*/
/* -------------------------------------------------------*/
/* --------------------Wellcome text effect-------------------

var words = [Welcome to automl],
    part,
    i = 0,
    offset = 0,
    len = words.length,
    forwards = true,
    skip_count = 0,
    skip_delay = 15,
    speed = 70;
var wordflick = function () {
  setInterval(function () {
    if (forwards) {
      if (offset >= words[i].length) {
        ++skip_count;
        if (skip_count == skip_delay) {
          forwards = false;
          skip_count = 0;
        }
      }
    }
    else {
      if (offset == 0) {
        forwards = true;
        i++;
        offset = 0;
        if (i >= len) {
          i = 0;
        }
      }
    }
    part = words[i].substr(0, offset);
    if (skip_count == 0) {
      if (forwards) {
        offset++;
      }
      else {
        offset--;
      }
    }
    $('.word').text(part);
  },speed);
};

$(document).ready(function () {
  wordflick();
});
-------*/
/* -------------------------------------------------------*/
/* -------------------------------------------------------*/
/* ----------------------------------------------------------------*/
/* -------------------------------------------------------*/
/* --------------------footerlinks---------------------------------*/
/* ----------------------------------------------------------------
*/
footertag = document.querySelector(".footerLinks");
var linktag1 = document.createElement("a"); 
var icontag1 = document.createElement("i"); 
icontag1.setAttribute("class", "fa fa-github");
linktag1.setAttribute("href", "https://github.com/Rafsun001");
linktag1.appendChild(icontag1);
footertag.appendChild(linktag1)

/* ----------------------------------------------------------------
*/
footertag = document.querySelector(".footerLinks");
var linktag2 = document.createElement("a"); 
var icontag2 = document.createElement("i"); 
icontag2.setAttribute("class", "fa fa-envelope");
linktag2.setAttribute("href", "mailto:ahmadrafsun001@gmail.com");
linktag2.appendChild(icontag2);
footertag.appendChild(linktag2)
/* ----------------------------------------------------------------
*/
footertag = document.querySelector(".footerLinks");
var linktag3 = document.createElement("a"); 
var icontag3 = document.createElement("i"); 
icontag3.setAttribute("class", "fa fa-linkedin");
linktag3.setAttribute("href", "https://www.linkedin.com/in/rafsun-ahmad-3222051b9/");
linktag3.appendChild(icontag3);
footertag.appendChild(linktag3)
/* ----------------------------------------------------------------
*/
footertag = document.querySelector(".footerLinks");
var linktag4 = document.createElement("a"); 
var icontag4 = document.createElement("i"); 
icontag4.setAttribute("class", "fa fa-instagram");
linktag4.setAttribute("href", "https://www.instagram.com/rafsunking001");
linktag4.appendChild(icontag4);
footertag.appendChild(linktag4)


