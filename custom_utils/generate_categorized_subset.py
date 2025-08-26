import shutil

def generate_categorized_subset(excel_path, images_source_folder, filename_column="file_name", output_folder="organized_images"):
    os.makedirs(output_folder, exist_ok=True)

    excel_data = pd.ExcelFile(excel_path)

    for sheet_name in excel_data.sheet_names:
        df = excel_data.parse(sheet_name)
        category_folder = os.path.join(output_folder, sheet_name)
        os.makedirs(category_folder, exist_ok=True)

        for _, row in df.iterrows():
            file_name = row[filename_column]
            src_path = os.path.join(images_source_folder, file_name)
            dst_path = os.path.join(category_folder, file_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: {src_path} not found.")

    print(f"Images organized into categories at: '{output_folder}'")

# Usage example:
data_path = '../../data/VisDrone2COCO/COCO/annotations/visdrone_coco_test.json'

excel_path = os.path.join("output_excel", "categorized_images.xlsx")
images_source_folder = '../../data/VisDrone2COCO/COCO/test2017/images'
generate_categorized_subset(excel_path, images_source_folder, filename_column="file_name", output_folder="categorized_images")
