all: run

run:
	uv run nsdget

publish:
	rm -fr dist/

	uv build
	uv publish --token $(TOKEN) 

	rm -fr dist/

download-betas:
	aws s3 sync s3://natural-scenes-dataset/nsddata_betas/ppdata ppdata/ --exclude "*" --include "*func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session**.nii.gz"

download-betas-fast:
	s5cmd sync --include "*func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session**.nii.gz" 's3://natural-scenes-dataset/nsddata_betas/ppdata/*' data/