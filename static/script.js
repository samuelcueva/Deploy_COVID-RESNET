$(window).load(function(){

 $(function() {
	 
  $('#imgSalida').hide() 
  $('#imageinput').change(function(e) {
      addImage(e);
     });

     function addImage(e){
      var file = e.target.files[0],
      imageType = /image.*/;

      if (!file.type.match(imageType))
       return;

      var reader = new FileReader();
      reader.onload = fileOnload;
      reader.readAsDataURL(file);
     }

     function fileOnload(e) {
      var result=e.target.result;
      $('#imgSalida').show();
      $('#imgSalida').attr("src",result);
     }

     
	 
     $('#imageinput').click(function(){
     	$('#div_result').hide()
     
     });


     $('form').on('submit', function(event) {
        var formData = new FormData(this);
		$.ajax({
			data : formData,
			type : 'POST',
			url : '/predict',
			contentType: false,
            processData: false
		})
		.done(function(data) {
            $('#div_result').show()
			$('#div_result').text('The x-ray result is: '+data.prediction+ ' with a '+data.percentage+'% probability')


		});

		event.preventDefault();

	});

    });
  });
