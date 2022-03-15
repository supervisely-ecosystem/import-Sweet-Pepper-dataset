
import zipfile, os, random
import supervisely as sly
import sly_globals as g
from supervisely.io.fs import get_file_name_with_ext, get_file_ext
from supervisely.io.json import load_json_file
from supervisely.imaging.image import read
import gdown


def prepare_ann_data(annotations_path):

    ann_json = load_json_file(annotations_path)

    for image_data in ann_json['_via_img_metadata'].values():
        polygons = []
        attributes = []
        for region in image_data['regions']:
            polygons.append(region['shape_attributes'])
            attributes.append(region['region_attributes'])
        g.image_name_to_polygon[image_data['filename']] = polygons
        g.image_name_to_attribute[image_data['filename']] = attributes


def create_ann(img_path):

    im = read(img_path)
    width = im.shape[1]
    height = im.shape[0]

    labels = []

    polygons = g.image_name_to_polygon[get_file_name_with_ext(img_path)]
    attributes = g.image_name_to_attribute[get_file_name_with_ext(img_path)]
    for index, poly in enumerate(polygons):
        x_points = poly['all_points_x']
        y_points = poly['all_points_y']
        points = []
        for idx in range(len(x_points)):
            points.append(sly.PointLocation(y_points[idx], x_points[idx]))

        polygon = sly.Polygon(points, interior=[])

        tag_color = sly.Tag(g.tag_meta_color, value=attributes[index]['Color'])
        tag_type = sly.Tag(g.tag_meta_type, value=attributes[index]['Type'])

        label = sly.Label(polygon, g.obj_class, tags=sly.TagCollection([tag_color, tag_type]))
        labels.append(label)

    return sly.Annotation(img_size=(height, width), labels=labels)


def extract_zip():
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


@g.my_app.callback("import_strawberry")
@sly.timeit
def import_strawberry(api: sly.Api, task_id, context, state, app_logger):

    gdown.download(g.strawberry_url, g.archive_path, quiet=False)
    extract_zip()

    items_path = os.path.join(g.work_dir_path, g.extract_folder_name)
    images_names = [image_name for image_name in os.listdir(items_path) if get_file_ext(image_name) == g.images_ext]
    annotations_path = os.path.join(items_path, g.annotation_file_name)
    prepare_ann_data(annotations_path)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    new_dataset = api.dataset.create(new_project.id, g.dataset_name, change_name_if_conflict=True)

    progress = sly.Progress('Upload items', len(items_path), app_logger)

    sample_img_names = random.sample(images_names, g.sample_percent)

    for img_batch in sly.batched(sample_img_names, batch_size=g.batch_size):
        img_pathes = [os.path.join(items_path, img_name) for img_name in img_batch]
        img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(img_path) for img_path in img_pathes]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_strawberry"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)