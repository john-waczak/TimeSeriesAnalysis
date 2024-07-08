set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)





import StatisticalMeasures: RSquared

rsq(ŷ, y) = RSquared()(ŷ, y)



function scatter_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ga = fig[1, 1] = GridLayout()
    axtop = Axis(ga[1, 1];
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
                )
    axmain = Axis(ga[2, 1], xlabel = "True $(varname)", ylabel = "Predicted $(varname)")
    axright = Axis(ga[2, 2];
                  leftspinevisible = false,
                  rightspinevisible = false,
                  bottomspinevisible = false,
                  topspinevisible = false,
                  )

    linkyaxes!(axmain, axright)
    linkxaxes!(axmain, axtop)

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    s1 = scatter!(axmain, y, ŷ, alpha=0.75)
    s2 = scatter!(axmain, ytest, ŷtest, marker=:rect, alpha=0.75)

    labels=[
        "Training R²=$(round(rsq(ŷ, y), digits=3)) (n=$(length(y)))",
        "Testing   R²=$(round(rsq(ŷtest, ytest), digits=3)) (n=$(length(ytest)))",
        "1:1"
    ]

    # leg = Legend(ga[1, 2], [s1, s2, l1], labels)
    leg = axislegend(axmain, [s1, s2, l1], labels; position=:lt)

    density!(axtop, y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axtop, ytest, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    density!(axright, ŷ, direction = :y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axright, ŷtest, direction = :y, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    hidedecorations!(axtop)
    hidedecorations!(axright)
    #leg.tellheight = true
    rowsize!(ga, 1, Relative(0.1))
    colsize!(ga, 2, Relative(0.1))

    colgap!(ga, 0)
    rowgap!(ga, 0)

    xlims!(axmain, minval-δ_edge, maxval+δ_edge)
    ylims!(axmain, minval-δ_edge, maxval+δ_edge)


    return fig
end




function quantile_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="True $(varname)", ylabel="Predicted $(varname)")

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(ax, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    qtrain = qqplot!(ax, y, ŷ, alpha=0.5)
    qtest = qqplot!(ax, ytest, ŷtest, marker=:rect, alpha=0.5)

    leg = axislegend(ax, [qtrain, qtest, l1], ["Training", "Testing", "1:1"]; position=:lt)

    return fig
end


