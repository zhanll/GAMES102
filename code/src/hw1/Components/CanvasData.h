#pragma once

#include <UGM/UGM.h>
#include "imgui/imgui.h"

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<ImVec2> LagrangeResults;
	std::vector<ImVec2> GaussResults;
	std::vector<ImVec2> LeastSquaresResults;
	std::vector<ImVec2> RidgeRegressionResults;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool opt_lagrange{ true };
	bool opt_gauss{ false };
	bool opt_least_squares{ false };
	bool opt_ridge_regression{ false };
	bool adding_line{ false };
	int LeastSquaresM = 4;
	float RidgeRegressionLamda = 0.1;
};

#include "details/CanvasData_AutoRefl.inl"
