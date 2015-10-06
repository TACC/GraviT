GraviT provides two benefits to the user:
	* performant, platform-independent ray tracing
	* scalable rendering across distributed architecture using in-core and out-of-core processing, as required

To use GraviT, an application must:
	* initialize GraviT
	* pass the data to be rendered to GraviT. This can be done by explicitly defining it, loading it from file, or passing it through the interface
	* define scene parameters, such as camera, lights and image buffer size
	* define the work scheduler to be used

GraviT program state is maintained through an object database. Each database node has a unique identifierThe database structure is described below:
	Mesh
		file (string) - filename for data
		bbox (Box3D) - data extent in world coordinates
		ptr (Mesh*) - pointer to mesh (gvt::render::data::primitives::Mesh)
	Instances
		Instance
			id (int) - instance id
			meshRef (UUID) - reference to Mesh 
			mat (AffineTransformMatrix) - transformation matrix for instance
			matInv (AffineTransformMatrix) - transformation matrix inverse (stored seperately for efficiency)
			normi (Matrix3f) - inverse transpose of transformation matrix
			bbox (Box3D) - data extent in world coordinates
			centroid (Point4f) - centroid for data instance, in global coordinates
	Lights
		PointLight
			position (Point4f) - light position in world coordinates
			color (Vector4f) - light color in RGB format
	Camera
		eyePoint (Point4f) - camera location in world coordinates
		focus (Point4f) - camera target in world coordinates
		upVector (Vector4f) - orientation vector for camera
		fov (float) - camera field of view in radians
	Film
		width (int) - width of image buffer in pixels
		height (int) - height of image buffer in pixels
	Schedule 
		type (gvt::render::scheduler class) - work scheduler, currently either Image or Domain
		adapter (gvt::render::adapter class) - ray tracing engine adapter, currently one of Manta, Embree or Optix
		