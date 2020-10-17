#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>
#include "../Eigen/Core"
#include "../Eigen/Dense"

using namespace Ubpa;

float PAI(const std::vector<Ubpa::pointf2>& P, int j, float x)
{
	int n = P.size();
	float numerator = 1;
	float denominator = 1;
	for (int k=0; k<n; ++k)
	{
		if (j == k)
		{
			continue;
		}
		numerator *= (x - P[k][0]);
		denominator *= (P[j][0] - P[k][0]);
	}
	return numerator / denominator;
}

float Polynomial(const std::vector<Ubpa::pointf2>& P, float x)
{
	int n = P.size();
	float sum = 0;
	for (int j=0; j<n; ++j)
	{
		sum += P[j][1] * PAI(P, j, x);
	}
	return sum;
}

float Gauss(const std::vector<Ubpa::pointf2>& P, float x)
{
	int n = P.size();
	if (n == 0)
	{
		return 0;
	}
	float theta = 100;
	Eigen::MatrixXf A(n, n);
	for (int row=0; row<n; ++row)
	{
		for (int col=0; col<n; ++col)
		{
			A(row, col) = (std::exp( - (P[row][0] - P[col][0]) * (P[row][0] - P[col][0]) / (2*theta*theta) ));
		}
	}
	Eigen::VectorXf b(n);
	for (int i=0; i<n; ++i)
	{
		b(i) = P[i][1];
	}
	Eigen::VectorXf a = A.colPivHouseholderQr().solve(b);
	float result = 0;
	for (int j=0; j<n; ++j)
	{
		result += a[j] * (std::exp( - (x - P[j][0]) * (x - P[j][0]) / (2*theta*theta) ));
	}
	return result;
}

float LeastSquares(const std::vector<Ubpa::pointf2>& P, float x, int m)
{
	int n = P.size();
	if (n == 0)
	{
		return 0;
	}
	if (m >= n)
	{
		m = n - 1;
	}
	Eigen::MatrixXf X(n, m);
	for (int i=0; i<n; ++i)
	{
		for (int j=0; j<m; ++j)
		{
			X(i, j) = std::powf(P[i][0], j);
		}
	}
	Eigen::VectorXf Y(n);
	for (int i=0; i<n; ++i)
	{
		Y(i) = P[i][1];
	}
	Eigen::VectorXf Theta = (X.transpose() * X).inverse() * X.transpose() * Y;
	//Eigen::VectorXf Theta = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
	float result = 0;
	for (int j=0; j<m; ++j)
	{
		result += Theta[j] * std::powf(x, j);
	}
	return result;
}

float RidgetRegression(const std::vector<Ubpa::pointf2>& P, float x, float lamda, int m)
{
	int n = P.size();
	if (n == 0)
	{
		return 0;
	}
	if (m >= n)
	{
		m = n - 1;
	}
	Eigen::MatrixXf X(n, m);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			X(i, j) = std::powf(P[i][0], j);
		}
	}
	Eigen::VectorXf Y(n);
	for (int i = 0; i < n; ++i)
	{
		Y(i) = P[i][1];
	}
	Eigen::MatrixXf I(m, m);
	I.setIdentity();
	Eigen::VectorXf Theta = (X.transpose() * X + I*lamda).inverse() * X.transpose() * Y;
	float result = 0;
	for (int j = 0; j < m; ++j)
	{
		result += Theta[j] * std::powf(x, j);
	}
	return result;
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: click to add points,\nMouse Right: drag to scroll, click for context menu.");

			ImGui::Checkbox("Lagrange", &data->opt_lagrange);
			ImGui::Checkbox("Gauss", &data->opt_gauss);

			ImGui::Checkbox("LeastSquares", &data->opt_least_squares);
			ImGui::SameLine(200);
			ImGui::InputInt("m", &data->LeastSquaresM);

			ImGui::Checkbox("RidgetRegression", &data->opt_ridge_regression);
			ImGui::SameLine(200);
			ImGui::InputFloat("lamda", &data->RidgeRegressionLamda, 0.01, 1, 3);

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			// Add first and second point
			if (is_hovered /*&& !data->adding_line*/ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);
				//data->points.push_back(mouse_pos_in_canvas);
				//data->adding_line = true;
				float Xmin = 999999;
				float Xmax = -99999;
				for (size_t i=0; i<data->points.size(); ++i)
				{
					if (data->points[i][0] < Xmin)
					{
						Xmin = data->points[i][0];
					}
					if (data->points[i][0] > Xmax)
					{
						Xmax = data->points[i][0];
					}
				}
				data->LagrangeResults.clear();
				data->GaussResults.clear();
				data->LeastSquaresResults.clear();
				data->RidgeRegressionResults.clear();
				for (int x=Xmin-1; x<Xmax+2; ++x)
				{
					data->LagrangeResults.push_back(ImVec2(origin.x + x, origin.y + Polynomial(data->points, x)));
					data->GaussResults.push_back(ImVec2(origin.x + x, origin.y + Gauss(data->points, x)));
					data->LeastSquaresResults.push_back(ImVec2(origin.x + x, origin.y + LeastSquares(data->points, x, data->LeastSquaresM)));
					data->RidgeRegressionResults.push_back(ImVec2(origin.x + x, origin.y + RidgetRegression(data->points, x, data->RidgeRegressionLamda, data->LeastSquaresM)));
				}
			}
			/*if (data->adding_line)
			{
				data->points.back() = mouse_pos_in_canvas;
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->adding_line = false;
			}*/

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			/*if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}*/

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				/*if (data->adding_line)
					data->points.resize(data->points.size() - 2);
				data->adding_line = false;*/
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			for (int n = 0; n < data->points.size(); n++)
			{
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), 4.0f, IM_COL32(255, 255, 0, 255));
			}
			//for (int n = 0; n < data->points.size(); n += 2)
				//draw_list->AddLine(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), IM_COL32(255, 255, 0, 255), 2.0f);
			
			if (data->opt_lagrange)
			{
				draw_list->AddPolyline(data->LagrangeResults.data(), data->LagrangeResults.size(), IM_COL32(64, 128, 255, 255), false, 1.0f);
			}
			if (data->opt_gauss)
			{
				draw_list->AddPolyline(data->GaussResults.data(), data->GaussResults.size(), IM_COL32(128, 255, 255, 255), false, 1.0f);
			}
			if (data->opt_least_squares)
			{
				draw_list->AddPolyline(data->LeastSquaresResults.data(), data->LeastSquaresResults.size(), IM_COL32(255, 128, 128, 255), false, 1.0f);
			}
			if (data->opt_ridge_regression)
			{
				draw_list->AddPolyline(data->RidgeRegressionResults.data(), data->RidgeRegressionResults.size(), IM_COL32(255	, 64, 64, 255), false, 1.0f);
			}
			
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
