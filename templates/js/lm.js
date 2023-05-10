var testMenu=[
    {"name": "图像识别与分割",
      "submenu": [
            {"name": "图像识别",
                "submenu": [
                   {"name": "单张识别","url": "  {{ url_for('image')}}  "},
				   {"name": "批量识别","url": ""}]},
            {"name": "图像分割",
                "submenu": [
                   {"name": "单张分割","url": ""},
				   {"name": "批量分割","url": ""},]}]
	}
];
	$(function(){
		new AccordionMenu({menuArrs:testMenu});
	});