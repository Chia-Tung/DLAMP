import logging
import tarfile
from pathlib import Path

logging.basicConfig(
    filename=Path(__file__).parent.resolve() / "unzip.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main():
    logger = logging.getLogger("dev")
    logger.info("start to unzip")

    file_dir = Path("/wk1/rwf/")
    tar_gz_files = list(file_dir.glob("*.tar.gz"))

    for tar_gz_file in tar_gz_files[-2:]:
        # new dir
        new_dir_name = tar_gz_file.name.split(".")[0]
        new_dir = Path(tar_gz_file.parent / new_dir_name)
        new_dir.mkdir(parents=True, exist_ok=True)

        # extraction
        tar_gz_file = str(tar_gz_file)
        new_dir = str(new_dir)
        try:
            if tar_gz_file.endswith("tar.gz"):
                tar = tarfile.open(tar_gz_file, "r:gz")
                tar.extractall(new_dir)
                tar.close()
            elif tar_gz_file.endswith("tar"):
                tar = tarfile.open(tar_gz_file, "r:")
                tar.extractall(new_dir)
                tar.close()
        except Exception as e:
            logger.error(e)

        # done
        logger.info(f"{tar_gz_file} has been extracted to {new_dir}")


def move_files():
    file_dir = Path("/wk1/rwf/")
    target_subdir = ["rwf_202005-06", "rwf_202105-06"]

    for subdir in target_subdir:
        subdir = file_dir / subdir

        for file in subdir.iterdir():
            new_dir = file_dir / f"rwf_{file.name[:6]}"
            if not new_dir.exists():
                new_dir.mkdir(parents=True, exist_ok=True)

            # move file to new dir
            file.rename(new_dir / file.name)

        print(f"{subdir} has been moved to {new_dir}")


if __name__ == "__main__":
    main()
    move_files()
