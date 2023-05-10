var testMenu=[
    {"name": "图像识别与分割",
      "submenu": [
            {"name": "图像识别",
                "submenu": [
                   {"name": "单张识别","url": "/image2"},
				   {"name": "批量识别","url": "/image3"},]},
            {"name": "图像分割",
                "submenu": [
                   {"name": "单张分割","url": "/image1"},
				   {"name": "批量分割","url": "/image"},]}]
	}
];
	$(function(){
		new AccordionMenu({menuArrs:testMenu});
	});