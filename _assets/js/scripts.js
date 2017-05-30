//= require typed
//= require animatedModal
//= require dimple.v2.3.0
//= require parsley

$(document).ready(function() {

	// Content:
	// - Typed.js Intro
	// - Scroll to top
	// - Animated Modals
	// - D3/Dimple charts
	
  // Typed Intro 		
  $(function(){
  	$('#typed-intro').typed({
      stringsElement: document.getElementById('typed-strings'),
      startDelay: 2000,
      typeSpeed: 0,
      callback: function() {
        $(".typed-cursor").hide();
        introEnd();
      }
    });
  });

  function introEnd() {
    Typed.new('#intro-end', {
      strings: ["understand cause and effect.","extract decision making insight.","help machines learn.","make the world a better place."],
      backDelay: 1500,
      typeSpeed: 0,
      backSpeed: 0,
      callback: function() {
        $('.hm-item').addClass('animated pulse');
      }
    });
  }


  // Scroll to top
  $(".back-to-top i").click(function(){
    $('body,html').animate({
        scrollTop: 0
    }, 2000);
  });


  // Animated Modals
    $("#skillsAn").animatedModal({
      modalTarget: 'skillsModal',
      color: '#202020',
      animatedIn: 'slideInDown',
      animatedOut: 'slideOutUp',
      overflow: 'hidden',
      afterOpen: function() {
        $('#skillsModal').css("transform",'initial');
      },
      afterClose: function() {
        $('body').removeAttr('style');
      }
    });


    $("#contactAn").animatedModal({
      modalTarget: 'contactModal',
      color: '#202020',
      animatedIn: 'slideInDown',
      animatedOut: 'slideOutUp',
      overflow: 'hidden',
      afterOpen: function() {
        $('#contactModal').css("transform",'initial');
      },
      afterClose: function() {
        $('body').removeAttr('style');
      }
    });

    //D3/Dimple charts
    // Resetting Modals
  $(".modal-content").removeAttr('style');

    var svg = dimple.newSvg("#skillchart-container", "100%","100%");

    d3.select("html")
                .style("height", "100%")
                .style("overflow", "hidden");
    d3.select("body")
                .style("height", "100%")
                .style("overflow", "hidden");

    // // Skill Set #1
    var data = [
    { Skill: "Python", Level: 7},
    { Skill: "R", Level: 7},
    { Skill: "Scikit-learn", Level: 7},
    { Skill: "Numpy", Level:  6},
    { Skill: "Pandas", Level:  8},
    { Skill: "ML Algorithms", Level:  7},
    { Skill: "Statistics", Level:  7},
    { Skill: "SQL/NoSQL", Level:  8},
    ];

    var myChart = new dimple.chart(svg, data);
    myChart.setMargins("40px","40px","30px","90px");

    //Set default colour
    myChart.defaultColors = [
          new dimple.color("#176cc1"),
      ];


    // myChart.setBounds(60, 30, 510, 305)
    var x = myChart.addCategoryAxis("x", "Skill");
    x.addOrderRule(["Python", "R", "Pandas","Numpy","ML Algorithms","SQL/NoSQL"]); //Set order

    var y = myChart.addMeasureAxis("y", "Level");
    y.overrideMin = 0;
    y.overrideMax = 10;

    myChart.addSeries(null, dimple.plot.bar);

    myChart.draw();

    window.onresize = function() {
      myChart.draw(0,true);
    }

    $("input[name='skillSelect']:radio").on("change", function(){
      if (this.value=='radioSE') {
    
    //Ugly hack to redraw chart
    d3.select('svg').remove();

    var svg = dimple.newSvg("#skillchart-container", "100%","100%");

    d3.select("html")
                .style("height", "100%")
                .style("overflow", "hidden");
    d3.select("body")
                .style("height", "100%")
                .style("overflow", "hidden");

    // // Skill Set #2
    var data = [
          { Skill: "HTML", Level:  9},
          { Skill: "CSS", Level:  7},
          { Skill: "Bootstrap", Level:  8},
          { Skill: "JS/Jquery", Level:  7},
          { Skill: "Java", Level:  6},
          { Skill: "Ruby", Level:  6},
          { Skill: "SQL/NoSQL", Level:  8},
          { Skill: "Python", Level:  7}
        ];

    var myChart = new dimple.chart(svg, data);
    myChart.setMargins("40px","40px","30px","90px");

    //Set default colour
    myChart.defaultColors = [
          new dimple.color("#b71919"),
      ];

    // myChart.setBounds(60, 30, 510, 305)
    var x = myChart.addCategoryAxis("x", "Skill");
    x.addOrderRule(["HTML", "CSS", "JS/Jquery","Bootstrap","Java","Ruby","Python","SQL/NoSQL"]); //Set order 
    var y = myChart.addMeasureAxis("y", "Level");
    y.overrideMin = 0;
    y.overrideMax = 10;
    myChart.addSeries(null, dimple.plot.bar);
    myChart.draw(1000);

    } 

    else {

    //Ugly hack to redraw chart
    d3.select('svg').remove();

    var svg = dimple.newSvg("#skillchart-container", "100%","100%");

    d3.select("html")
                .style("height", "100%")
                .style("overflow", "hidden");
    d3.select("body")
                .style("height", "100%")
                .style("overflow", "hidden");

    // // Skill Set #1
    var data = [
    { Skill: "Python", Level: 7},
    { Skill: "R", Level: 6},
    { Skill: "Scikit-learn", Level: 7},
    { Skill: "Numpy", Level:  6},
    { Skill: "Pandas", Level:  8},
    { Skill: "ML Algorithms", Level:  7},
    { Skill: "Statistics", Level:  7},
    { Skill: "SQL/NoSQL", Level:  8},
    ];

    var myChart = new dimple.chart(svg, data);
    myChart.setMargins("40px","40px","30px","90px");

    //Set default colour
    myChart.defaultColors = [
          new dimple.color("#176cc1"),
      ];

    // myChart.setBounds(60, 30, 510, 305)
    var x = myChart.addCategoryAxis("x", "Skill");
    x.addOrderRule(["Python", "R", "Pandas","Numpy","ML Algorithms","SQL/NoSQL"]); //Set order
    var y = myChart.addMeasureAxis("y", "Level");
    y.overrideMin = 0;
    y.overrideMax = 10;
    myChart.addSeries(null, dimple.plot.bar);
    myChart.draw(1000);
      
    }
    });



    // Skip Intro
    // $('#hello-world').click(function() {
    //   // Hiding the typedjs intro content
    //   $('#typed-intro').hide();
    //   $('#intro-end').hide();
    //   $(".typed-cursor").hide();

    //   //Displaying standard content
    //   // $('#skipped-intro').show();
    //   $('.hm-item').addClass('animated pulse');
    // });

});