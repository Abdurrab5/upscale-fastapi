from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    BackgroundTasks
)

from fastapi.responses import (
    FileResponse,
    JSONResponse
)

import shutil
import traceback

from services.upscale_service import upscale_image

from utils.image import (
    create_input_path,
    create_output_path,
    cleanup
)

from utils.progress import (
    update_progress,
    get_progress,
    remove_progress
)

router = APIRouter()


@router.get("/progress/{job_id}")
def progress(job_id: str):
    return get_progress(job_id)


@router.post("/upscale")
async def upscale(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_id: str = Form(...)
):

    input_path = create_input_path(file.filename)
    output_path = create_output_path()

    try:

        update_progress(
            job_id,
            5,
            "Uploading image"
        )

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        update_progress(
            job_id,
            15,
            "Preparing image"
        )

        await upscale_image(
            input_path,
            output_path,
            job_id
        )

        background_tasks.add_task(
            cleanup,
            input_path,
            output_path
        )

        background_tasks.add_task(
            remove_progress,
            job_id
        )

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="upscaled.png"
        )

    except Exception as e:

        print(traceback.format_exc())

        update_progress(
            job_id,
            0,
            "Failed"
        )

        cleanup(
            input_path,
            output_path
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )