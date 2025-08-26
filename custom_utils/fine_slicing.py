def predict_fine_sliced_images(input_folder, dataset_json_path, detection_model, base_slice_size=512):
    name = "exp"
    save_dir = Path(increment_path(Path("sliced_predictions") / name, exist_ok=False))
    os.makedirs(save_dir, exist_ok=True)

    vis_params = {
        "bbox_thickness": 2,
        "text_size": 0.5,
        "text_thickness": 1,
        "hide_labels": False,
        "hide_conf": False,
        "format": "png"
    }

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n🚀 Running fine slicing prediction on {len(image_files)} images...")
    total_prediction_time = 0.0

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        image_h, image_w = image_np.shape[:2]
        filename_wo_ext = Path(filename).stem
        total_prediction_count = 0
        print("*****************************************")
        print("File Name", filename_wo_ext)

        # Split image into 2x2 grid
        grid_h, grid_w = image_h // 2, image_w // 2

        for row in range(2):
            for col in range(2):
                x1, y1 = col * grid_w, row * grid_h
                x2, y2 = min(x1 + grid_w, image_w), min(y1 + grid_h, image_h)
                sub_img = image_pil.crop((x1, y1, x2, y2))
                print("Cropped Image:", x1, y1, x2, y2)

                # Get initial predictions from your detection model
                time_start = time.time()
                # Base prediction on the sub-image
                base_pred = get_prediction(sub_img, detection_model)
                time_end = time.time() - time_start
                print("Initial Prediction time is: {:.2f} ms".format(time_end * 1000))
                
                object_density = len(base_pred.object_prediction_list)
                print("Object Density:", object_density)
                
                slice_params = get_slice_parameters(object_density, base_slice_size)

                # Add initial prediction time to the cumulative total.
                iteration_time = time_end
                
                if slice_params:
                    slice_width, slice_height, overlap_w, overlap_h = slice_params

                    print("********* Slice Parameters ***********")
                    print("Slice Width: ", slice_width)
                    print("Slice Height: ", slice_height)
                    print("Overlap Width Ratio: ", overlap_w)
                    print("Overlap Height Ratio: ", overlap_h)

                    time_start_slice = time.time()
                    sliced_pred = get_sliced_prediction(
                        sub_img,
                        detection_model,
                        slice_height=slice_height,
                        slice_width=slice_width,
                        overlap_height_ratio=overlap_h,
                        overlap_width_ratio=overlap_w,
                        postprocess_type="OptNMS",
                        postprocess_match_metric="IOU",
                        postprocess_match_threshold=0.3,
                        postprocess_min_area=16,
                        verbose=0
                    )
                    time_end_slice = time.time() - time_start_slice
                    print("Sliced Prediction time is: {:.2f} ms".format(time_end_slice * 1000))
                
                    # Add sliced prediction time to the current iteration's total.
                    iteration_time += time_end_slice
                
                    preds = sliced_pred.object_prediction_list
                    total_prediction_count += len(preds)
                    print("Sliced Level Prediction Count 1: ", len(preds))
                else:
                    print("Prediction time is: {:.2f} ms".format(time_end * 1000))
                    preds = base_pred.object_prediction_list
                    total_prediction_count += len(preds)
                    print("Sliced Level Prediction Count 2: ", len(preds))


                # Visualization for the current slice
                slice_filename = f"{filename_wo_ext}_slice_r{row}_c{col}"
                visualize_object_predictions(
                    image=np.ascontiguousarray(sub_img),
                    object_prediction_list=preds,
                    rect_th=vis_params["bbox_thickness"],
                    text_size=vis_params["text_size"],
                    text_th=vis_params["text_thickness"],
                    hide_labels=vis_params["hide_labels"],
                    hide_conf=vis_params["hide_conf"],
                    output_dir=save_dir,
                    file_name=slice_filename,
                    export_format=vis_params["format"]
                )
                # Update the overall total prediction time
                total_prediction_time += iteration_time
    
    print(f"\n✅ Completed {len(image_files)} images.")
    print("Total Prediction Count: ", (total_prediction_count))
    print("Total Prediction time for all images is: {:.2f} ms".format(total_prediction_time * 1000))
    print(f"Prediction results are successfully exported to {save_dir}")
